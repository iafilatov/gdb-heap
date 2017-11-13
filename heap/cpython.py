'''
This file is licensed under the PSF license
'''
import re
import sys

import gdb
from heap import WrappedPointer, caching_lookup_type, Usage, \
    type_void_ptr, fmt_addr, Category, looks_like_ptr, \
    WrongInferiorProcess, Table

import libpython

from libpython import (
    Py_TPFLAGS_HEAPTYPE,
    Py_TPFLAGS_LONG_SUBCLASS,
    Py_TPFLAGS_LIST_SUBCLASS,
    Py_TPFLAGS_TUPLE_SUBCLASS,
    Py_TPFLAGS_BYTES_SUBCLASS,
    Py_TPFLAGS_UNICODE_SUBCLASS,
    Py_TPFLAGS_DICT_SUBCLASS,
    Py_TPFLAGS_BASE_EXC_SUBCLASS,
    Py_TPFLAGS_TYPE_SUBCLASS,
    _PyObject_VAR_SIZE,
    int_from_int,
    PyTypeObjectPtr,
)


SIZEOF_VOID_P = type_void_ptr.sizeof

# Transliteration from Python's obmalloc.c:
ALIGNMENT             = 8
ALIGNMENT_SHIFT       = 3
ALIGNMENT_MASK        = (ALIGNMENT - 1)

# Return the number of bytes in size class I:
def INDEX2SIZE(I):
    return (I + 1) << ALIGNMENT_SHIFT

SYSTEM_PAGE_SIZE      = (4 * 1024)
SYSTEM_PAGE_SIZE_MASK = (SYSTEM_PAGE_SIZE - 1)
ARENA_SIZE            = (256 << 10)
POOL_SIZE             = SYSTEM_PAGE_SIZE
POOL_SIZE_MASK        = SYSTEM_PAGE_SIZE_MASK
def ROUNDUP(x):
    return (x + ALIGNMENT_MASK) & ~ALIGNMENT_MASK

def POOL_OVERHEAD():
    return ROUNDUP(caching_lookup_type('struct pool_header').sizeof)

class PyArenaPtr(WrappedPointer):
    # Wrapper around a (void*) that's a Python arena's buffer (the
    # arena->address, as opposed to the (struct arena_object*) itself)
    @classmethod
    def from_addr(cls, p, arenaobj):
        ptr = gdb.Value(p)
        ptr = ptr.cast(type_void_ptr)
        return cls(ptr, arenaobj)

    def __init__(self, gdbval, arenaobj):
        WrappedPointer.__init__(self, gdbval)

        assert(isinstance(arenaobj, ArenaObject))
        self.arenaobj = arenaobj

        # obmalloc.c sets up arenaobj->pool_address to the first pool
        # address, aligning it to POOL_SIZE_MASK:
        self.initial_pool_addr = self.as_address()
        self.num_pools = ARENA_SIZE // POOL_SIZE
        self.excess = self.initial_pool_addr & POOL_SIZE_MASK
        if self.excess != 0:
            self.num_pools -= 1
            self.initial_pool_addr += POOL_SIZE - self.excess

    def __str__(self):
        return ('PyArenaPtr([%s->%s], %i pools [%s->%s], excess: %i tracked by %s)'
                % (fmt_addr(self.as_address()),
                   fmt_addr(self.as_address() + ARENA_SIZE - 1),
                   self.num_pools,
                   fmt_addr(self.initial_pool_addr),
                   fmt_addr(self.initial_pool_addr
                            + (self.num_pools * POOL_SIZE) - 1),
                   self.excess,
                   self.arenaobj
                   )
                )

    def iter_pools(self):
        '''Yield a sequence of PyPoolPtr, representing all of the pools within
        this arena'''
        # print 'num_pools:', num_pools
        pool_addr = self.initial_pool_addr
        for idx in range(self.num_pools):

            # "pool_address" is a high-water-mark for activity within the arena;
            # pools at this location or beyond haven't been initialized yet:
            if pool_addr >= self.arenaobj.pool_address:
                return

            pool = PyPoolPtr.from_addr(pool_addr)
            yield pool
            pool_addr += POOL_SIZE

    def iter_usage(self):
        '''Yield a series of Usage instances'''
        if self.excess != 0:
            # FIXME: this size is wrong
            yield Usage(self.as_address(), self.excess, Category('pyarena', 'alignment wastage'))

        for pool in self.iter_pools():
            # print 'pool:', pool
            for u in pool.iter_usage():
                yield u

        # FIXME: unused space (if any) between pool_address and the alignment top

        # if self.excess != 0:
        #    # FIXME: this address is wrong
        #    yield Usage(self.as_address(), self.excess, Category('pyarena', 'alignment wastage'))


class PyPoolPtr(WrappedPointer):
    # Wrapper around Python's obmalloc.c: poolp: (struct pool_header *)

    @classmethod
    def from_addr(cls, p):
        ptr = gdb.Value(p)
        ptr = ptr.cast(cls.gdb_type())
        return cls(ptr)

    def __str__(self):
        return ('PyPoolPtr([%s->%s: %d blocks of size %i bytes))'
                % (fmt_addr(self.as_address()), fmt_addr(self.as_address() + POOL_SIZE - 1),
                   self.num_blocks(), self.block_size()))

    @classmethod
    def gdb_type(cls):
        # Deferred lookup of the "poolp" type:
        return caching_lookup_type('poolp')

    def block_size(self):
        return INDEX2SIZE(self.field('szidx'))

    def num_blocks(self):
        firstoffset = self._firstoffset()
        maxnextoffset = self._maxnextoffset()
        offsetrange = maxnextoffset - firstoffset
        return offsetrange / self.block_size() # FIXME: not exactly correctly

    def _firstoffset(self):
        return POOL_OVERHEAD()

    def _maxnextoffset(self):
        return POOL_SIZE - self.block_size()

    def iter_blocks(self):
        '''Yield all blocks within this pool, whether free or in use'''
        size = self.block_size()
        maxnextoffset = self._maxnextoffset()
        # print initnextoffset, maxnextoffset
        offset = self._firstoffset()
        base_addr = self.as_address()
        while offset <= maxnextoffset:
            yield (base_addr + offset, size)
            offset += size

    def iter_usage(self):
        # The struct pool_header at the front:
        yield Usage(self.as_address(),
                    POOL_OVERHEAD(),
                    Category('pyarena', 'pool_header overhead'))

        fb = list(self.iter_free_blocks())
        for (start, size) in fb:
            yield Usage(start, size, Category('pyarena', 'freed pool chunk'))

        for (start, size) in self.iter_used_blocks():
            if (start, size) not in fb:
                yield Usage(start, size) #, 'python pool: ' + categorize(start, size, None))

        # FIXME: yield any wastage at the end

    def iter_free_blocks(self):
        '''Yield the sequence of free blocks within this pool.  Doesn't include
        the areas after nextoffset that have never been allocated'''
        # print self._gdbval.dereference()
        size = self.block_size()
        freeblock = self.field('freeblock')
        _type_block_ptr_ptr = caching_lookup_type('unsigned char').pointer().pointer()
        # Walk the singly-linked list of free blocks for this chunk
        while int(freeblock) != 0:
            # print 'freeblock:', (fmt_addr(int(freeblock)), int(size))
            yield (int(freeblock), int(size))
            freeblock = freeblock.cast(_type_block_ptr_ptr).dereference()

    def _free_blocks(self):
        # Get the set of addresses of free blocks
        return set([addr for addr, size in self.iter_free_blocks()])

    def iter_used_blocks(self):
        '''Yield the sequence of currently in-use blocks within this pool'''
        # We'll filter out the free blocks from the list:
        free_block_addresses = self._free_blocks()

        size = self.block_size()
        offset = self._firstoffset()
        nextoffset = self.field('nextoffset')
        base_addr = self.as_address()
        # Iterate upwards until you reach "pool->nextoffset": blocks beyond
        # that point have never been allocated:
        while offset < nextoffset:
            addr = base_addr + offset
            # Filter out those within this pool's linked list of free blocks:
            if int(addr) not in free_block_addresses:
                yield (int(addr), int(size))
            offset += size


class PyObjectPtr(libpython.PyObjectPtr, WrappedPointer):

    @classmethod
    def subclass_from_type(cls, t):
        # OMG what am I doing? Don't look.
        klass = super().subclass_from_type(t)

        def categorize(self):
            # Python objects will be categorized as ("python", tp_name), but
            # old-style classes have to do more work
            # l = [str(self)]
            # try:
            #     l.append(str(self.type()))
            #     l.append(self.type().field('tp_name').string())
            # except Exception as e:
            #     print('categorize')
            #     print('\n'.join(l))
            #     print(e)
            return Category('python', self.safe_tp_name())

        klass.categorize = categorize

        if klass is libpython.HeapTypeObjectPtr:
            return HeapTypeObjectPtr
        elif klass is libpython.PyDictObjectPtr:
            return PyDictObjectPtr
        # elif klass is libpython.PyUnicodeObjectPtr:
        #     return PyUnicodeObjectPtr
        # elif klass is libpython.PyLongObjectPtr:
        #     return PyLongObjectPtr
        # elif klass is libpython.PyListObjectPtr:
        #     return PyListObjectPtr
        # elif klass is libpython.PyTypeObjectPtr:
        #     return PyTypeObjectPtr
        # elif klass is libpython.PyBytesObjectPtr:
        #     return PyBytesObjectPtr
        # elif klass is libpython.PyBaseExceptionObjectPtr:
        #     return PyBaseExceptionObjectPtr
        return klass

    def as_malloc_addr(self):
        ob_type = addr['ob_type']
        tp_flags = ob_type['tp_flags']
        addr = int(self._gdbval)
        if tp_flags & Py_TPFLAGS_: # FIXME
            return obj_addr_to_gc_addr(addr)
        else:
            return addr

class PyUnicodeObjectPtr(libpython.PyUnicodeObjectPtr, PyObjectPtr):
    pass

class PyLongObjectPtr(libpython.PyLongObjectPtr, PyObjectPtr):
    pass

class PyListObjectPtr(libpython.PyListObjectPtr, PyObjectPtr):
    pass

class PyTupleObjectPtr(libpython.PyTupleObjectPtr, PyObjectPtr):
    pass

class PyBytesObjectPtr(libpython.PyBytesObjectPtr, PyObjectPtr):
    pass

class PyBaseExceptionObjectPtr(libpython.PyBaseExceptionObjectPtr, PyObjectPtr):
    pass

class PyDictObjectPtr(libpython.PyDictObjectPtr, PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyDictObject* i.e. a dict instance
    within the process being debugged.
    """
    _typename = 'PyDictObject'

    # FIXME: ma_keys isn't in usages so it's ingored anyway
    def categorize_refs(self, usage_set, level=0, detail=None):
        ma_keys = int(self.field('ma_keys'))
        usage_set.set_addr_category(ma_keys,
                                    Category('cpython', 'PyDictKeysObject', detail),
                                    level)
        return True

# FIXME: This is never used since we're using from_pyobject_ptr from libpython
class PyInstanceObjectPtr(PyObjectPtr):
    _typename = 'PyInstanceObject'

    def cl_name(self):
        in_class = self.field('in_class')
        # cl_name is a python string, not a char*; rely on
        # prettyprinters for now:
        cl_name = str(in_class['cl_name'])[1:-1]
        return cl_name

    def categorize(self):
        return Category('python', self.cl_name(), 'old-style')

    def categorize_refs(self, usage_set, level=0, detail=None):
        return True # FIXME
        cl_name = self.cl_name()
        # print 'cl_name', cl_name

        # Visit the in_dict:
        in_dict = self.field('in_dict')
        # print 'in_dict', in_dict

        dict_detail = '%s.__dict__' % cl_name

        # Mark the ptr as being a dictionary, adding detail
        usage_set.set_addr_category(obj_addr_to_gc_addr(in_dict),
                                    Category('cpython', 'PyDictObject', dict_detail),
                                    level=1)

        # Visit ma_table:
        _type_PyDictObject_ptr = caching_lookup_type('PyDictObject').pointer()
        in_dict = in_dict.cast(_type_PyDictObject_ptr)

        ma_table = int(in_dict['ma_table'])

        # Record details:
        usage_set.set_addr_category(ma_table,
                                    Category('cpython', 'PyDictEntry table', dict_detail),
                                    level=2)
        return True

class HeapTypeObjectPtr(libpython.HeapTypeObjectPtr, PyObjectPtr):
    _typename = 'PyObject'

    def categorize_refs(self, usage_set, level=0, detail=None):
        return True  # FIXME
        attr_dict = self.get_attr_dict()
        if attr_dict:
            # Mark the dictionary's "detail" with our typename
            # gdb.execute('print (PyObject*)0x%x' % int(attr_dict._gdbval))
            usage_set.set_addr_category(obj_addr_to_gc_addr(attr_dict._gdbval),
                                        Category('python', 'dict', '%s.__dict__' % self.safe_tp_name()),
                                        level=level+1)

            # and mark the dict's PyDictEntry with our typename:
            attr_dict.categorize_refs(usage_set, level=level+1,
                                      detail='%s.__dict__' % self.safe_tp_name())
        return True

def is_pyobject_ptr(addr):
    try:
        _type_pyop = caching_lookup_type('PyObject').pointer()
        _type_pyvarop = caching_lookup_type('PyVarObject').pointer()
    except RuntimeError:
        # not linked against python
        return None

    try:
        typeop = pyop = gdb.Value(addr).cast(_type_pyop)

        # If we follow type chain on a PyObject long enough, we should arrive
        # at 'type' and type(type) should be 'type'.
        # The levels are:     a <- b means a = type(b)
        # 0 - type
        # 1 - type <- class A
        #     type <- class M(type)
        # 2 - type <- class A <- A()
        #     type <- class M(type) <- class B(metaclass=M)
        # 3 - type <- class M(type) <- class B(metaclass=M) <- B()
        for i in range(4):
            if typeop['ob_type'] == typeop:
                return PyObjectPtr.from_pyobject_ptr(pyop)
            typeop = typeop['ob_type'].cast(_type_pyop)

        return 0

        # gdb.write('PYOP {}\n'.format(pyop))
        ob_refcnt = pyop['ob_refcnt']
        if ob_refcnt >=0 and ob_refcnt < 0xffff:
            # gdb.write('refcnt ok {}\n'.format(int(ob_refcnt)))
            obtype = pyop['ob_type']
            if looks_like_ptr(obtype):
                # gdb.write('obtype ok {}\n'.format(obtype))
                type_refcnt = obtype.cast(_type_pyop)['ob_refcnt']
                if type_refcnt > 0 and type_refcnt < 0xffff:
                    # gdb.write('type refcnt ok\n')
                    # type_ob_size = obtype.cast(_type_pyvarop)['ob_size']

                    # if type_ob_size > 0xffff:
                    #     return 0

                    # gdb.write('ob size ok\n')

                    for fieldname in ('tp_base', 'tp_free', 'tp_repr', 'tp_new'):
                        if not looks_like_ptr(obtype[fieldname]):
                            return 0

                    # gdb.write('methods ok\n')

                    # Then this looks like a Python object:
                    return PyObjectPtr.from_pyobject_ptr(pyop)

    except  UnicodeDecodeError as e:
        # print('is_pyobject_ptr 0x{:x}'.format(addr))
        # gdb.write(str(e) + '\n')
        pass

    except RuntimeError:
        pass # Not a python object (or corrupt)

    # Doesn't look like a python object, implicit return None

def obj_addr_to_gc_addr(addr):
    '''Given a PyObject* address, convert to a PyGC_Head* address
    (i.e. the allocator's view of the same)'''
    #print 'obj_addr_to_gc_addr(%s)' % fmt_addr(int(addr))
    _type_PyGC_Head = caching_lookup_type('PyGC_Head')
    return int(addr) - _type_PyGC_Head.sizeof

def as_python_object(addr):
    '''Given an address of an allocation, determine if it holds a PyObject,
    or a PyGC_Head

    Return a WrappedPointer for the PyObject* if it does (which might have a
    different location c.f. when PyGC_Head was allocated)

    Return None if it doesn't look like a PyObject*'''
    # Try casting to PyObject* ?
    # FIXME: what about the debug allocator?
    try:
        _type_pyop = caching_lookup_type('PyObject').pointer()
        _type_PyGC_Head = caching_lookup_type('PyGC_Head')
    except RuntimeError:
        # not linked against python
        return None
    pyop = is_pyobject_ptr(addr)
    if pyop:
        return pyop
    else:
        # maybe a GC type:
        _type_PyGC_Head_ptr = _type_PyGC_Head.pointer()
        gc_ptr = gdb.Value(addr).cast(_type_PyGC_Head_ptr)
        # print gc_ptr.dereference()

        # PYGC_REFS_REACHABLE = -3

        if gc_ptr['gc']['gc_refs'] in (-2, -3, -4):  # FIXME: need to cover other values
            pyop = is_pyobject_ptr(gdb.Value(addr + _type_PyGC_Head.sizeof))
            if pyop:
                return pyop
    # Doesn't look like a python object, implicit return None


class ArenaObject(WrappedPointer):
    '''
    Wrapper around Python's struct arena_object*
    Note that this is record-keeping for an arena, not the
    memory itself
    '''
    @classmethod
    def iter_arenas(cls):
        arenas_var, maxarenas_var = cls._get_arena_vars()
        try:
            val_arenas = gdb.parse_and_eval("'{}'".format(arenas_var))
            val_maxarenas = gdb.parse_and_eval("'{}'".format(maxarenas_var))
        except RuntimeError:
            # Not linked against python, or no debug information:
            raise WrongInferiorProcess('cpython')

        try:
            for i in range(int(val_maxarenas)):
                # Look up "&arenas[i]":
                obj = ArenaObject(val_arenas[i].address)

                # obj->address == 0 indicates an unused entry within the "arenas" array:
                if obj.address != 0:
                    yield obj
        except RuntimeError:
            # pypy also has a symbol named "arenas", of type "long unsigned int * volatile"
            # For now, ignore it:
            return

    @classmethod
    def _get_arena_vars(cls):
        '''
        Get names of 'arenas' and 'maxarenas' vars which can be chaged by
        link time optimization.
        '''
        # TODO: Is there a better way to do this?
        vars = gdb.execute('info variables arenas', False, True)
        arenas_pat = re.compile(r'struct arena_object \*(arenas(?:\.lto_priv\.\d+))')
        maxarenas_pat = re.compile(r'unsigned int (maxarenas(?:\.lto_priv\.\d+))')
        arenas = maxarenas = None
        for line in vars.splitlines():
            arenas_m = arenas_pat.match(line)
            if arenas_m:
                arenas = arenas_m.group(1)
            maxarenas_m = maxarenas_pat.match(line)
            if maxarenas_m:
                maxarenas = maxarenas_m.group(1)
            if arenas and maxarenas:
                return arenas, maxarenas

    @property  # need to override the base property
    def address(self):
        return self.field('address')

    def __init__(self, gdbval):
        WrappedPointer.__init__(self, gdbval)

        # Cache some values:
        # This is the high-water mark: at this point and beyond, the bytes of
        # memory are untouched since malloc:
        self.pool_address = self.field('pool_address')


class ArenaDetection(object):
    '''Detection of CPython arenas, done as an object so that we can cache state'''
    def __init__(self):
        self.arenaobjs = list(ArenaObject.iter_arenas())

    def as_arena(self, ptr, chunksize):
        '''Detect if this ptr returned by malloc is in use as a Python arena,
        returning PyArenaPtr if it is, None if not'''
        # Fast rejection of too-small chunks:
        # https://www.python.org/dev/peps/pep-0445/
        if chunksize < (256 * 1024):
            return None

        for arenaobj in self.arenaobjs:
            if ptr == arenaobj.address:
                # Found it:
                print('ARENA {:x} -> {:x}'.format(ptr, ptr + chunksize) )
                return PyArenaPtr.from_addr(ptr, arenaobj)

        # Not found:
        return None


def python_categorization(usage_set):
    # special-cased categorization for CPython

    # The Objects/stringobject.c:interned dictionary is typically large,
    # with its PyDictEntry table occuping 200k on a 64-bit build of python 2.6
    # Identify it:
    try:
        val_interned = gdb.parse_and_eval('interned')
        pyop = PyDictObjectPtr.from_pyobject_ptr(val_interned)
        ma_table = int(pyop.field('ma_table'))
        usage_set.set_addr_category(ma_table,
                                    Category('cpython', 'PyDictEntry table', 'interned'),
                                    level=1)
    except RuntimeError:
        pass

    # Various kinds of per-type optimized allocator
    # See Modules/gcmodule.c:clear_freelists

    # The Objects/intobject.c: block_list
    try:
        val_block_list = gdb.parse_and_eval('block_list')
        if str(val_block_list.type.target()) != 'PyIntBlock':
            raise RuntimeError
        while int(val_block_list) != 0:
            usage_set.set_addr_category(int(val_block_list),
                                        Category('cpython', '_intblock', ''),
                                        level=0)
            val_block_list = val_block_list['next']

    except RuntimeError:
        pass

    # The Objects/floatobject.c: block_list
    # TODO: how to get at this? multiple vars named "block_list"

    # Objects/methodobject.c: PyCFunction_ClearFreeList
    #   "free_list" of up to 256 PyCFunctionObject, but they're still of
    #   that type

    # Objects/classobject.c: PyMethod_ClearFreeList
    #   "free_list" of up to 256 PyMethodObject, but they're still of that type

    # Objects/frameobject.c: PyFrame_ClearFreeList
    #   "free_list" of up to 300 PyFrameObject, but they're still of that type

    # Objects/tupleobject.c: array of free_list: up to 2000 free tuples of each
    # size from 1-20 (using ob_item[0] to chain up); singleton for size 0; they
    # are still tuples when deallocated, though

    # Objects/unicodeobject.c:
    #   "free_list" of up to 1024 PyUnicodeObject, with the "str" buffer
    #   optionally preserved also for lengths up to 9
    #   They're all still of type "unicode" when free
    #   Singletons for the empty unicode string, and for the first 256 code
    #   points (Latin-1)

# New gdb commands, specific to CPython

from heap.commands import need_debuginfo


class HeapCPythonAllocators(gdb.Command):
    "For CPython: display information on the allocators"
    def __init__(self):
        gdb.Command.__init__ (self,
                              "heap cpython-allocators",
                              gdb.COMMAND_DATA)

    @need_debuginfo
    def invoke(self, args, from_tty):
        t = Table(columnheadings=('struct arena_object*', '256KB buffer location', 'Free pools'))
        for arena in ArenaObject.iter_arenas():
            t.add_row([fmt_addr(arena.as_address()),
                       fmt_addr(arena.address),
                       '%i / %i ' % (arena.field('nfreepools'),
                                     arena.field('ntotalpools'))
                       ])
        print('Objects/obmalloc.c: %i arenas' % len(t.rows))
        t.write(sys.stdout)
        print()


def register_commands():
    HeapCPythonAllocators()
