# $Id$
"""Atom selection Hierarchy

These objects are constructed and applied to the group

Currently all atom arrays are handled internally as sets, but returned as AtomGroups

"""

try:
    set([])
except NameError:
    from sets import Set as set

import numpy

from AtomGroup import AtomGroup, Universe
from MDAnalysis.core import flags


class Selection:
    def __init__(self):
        # This allows you to build a Selection without tying it to a particular group yet
        # Updatable means every timestep
        self.update = False   # not used at the moment
    def __repr__(self):
        return "<"+self.__class__.__name__+">"
    def __and__(self, other):
        return AndSelection(self, other)
    def __or__(self, other):
        return OrSelection(self, other)
    def __invert__(self):
        return NotSelection(self)
    def __hash__(self):
        return hash(repr(self))
    def _apply(self,group):
        # This is an error
        raise NotImplementedError("No _apply function defined for "+repr(self.__class__.__name__))
    def apply(self,group):
        # Cache the result for future use
        # atoms is from Universe
        # returns AtomGroup
        if not (isinstance(group, Universe) or isinstance(group,AtomGroup)):
            raise Exception("Must pass in an AtomGroup or Universe to the Selection")
        # make a set of all the atoms in the group
        # XXX this should be static to all the class members
        Selection._group_atoms = set(group.atoms)
        Selection._group_atoms_list = [a for a in Selection._group_atoms] # need ordered, unique list for back-indexing in Around and Point!
        if not hasattr(group, "coord"): Selection.coord = group.universe.coord
        else: Selection.coord = group.coord

        if not hasattr(self, "_cache"):
            cache = list(self._apply(group))
            # Decorate/Sort/Undecorate (Schwartzian Transform)
            cache[:] = [(x.number, x) for x in cache]
            cache.sort()
            cache[:] = [val for (key, val) in cache]
            self._cache = AtomGroup(cache)
        return self._cache

class AllSelection(Selection):
    def __init__(self):
        Selection.__init__(self)
    def _apply(self, group):
        return set(group.atoms[:])

class NotSelection(Selection):
    def __init__(self, sel):
        Selection.__init__(self)
        self.sel = sel
    def _apply(self, group):
        notsel = self.sel._apply(group)
        return (set(group.atoms[:])-notsel)
    def __repr__(self):
        return "<'NotSelection' "+repr(self.sel)+">"

class AndSelection(Selection):
    def __init__(self, lsel, rsel):
        Selection.__init__(self)
        self.rsel = rsel
        self.lsel = lsel
    def _apply(self, group):
        return self.lsel._apply(group) & self.rsel._apply(group)
    def __repr__(self):
        return "<'AndSelection' "+repr(self.lsel)+","+repr(self.rsel)+">"

class OrSelection(Selection):
    def __init__(self, lsel, rsel):
        Selection.__init__(self)
        self.rsel = rsel
        self.lsel = lsel
    def _apply(self, group):
        return self.lsel._apply(group) | self.rsel._apply(group)
    def __repr__(self):
        return "<'OrSelection' "+repr(self.lsel)+","+repr(self.rsel)+">"

class AroundSelection(Selection):
    def __init__(self, sel, cutoff, periodic=None):
        Selection.__init__(self)
        self.sel = sel
        self.cutoff = cutoff
        self.sqdist = cutoff*cutoff
        if periodic is None:
            self.periodic = flags['use_periodic_selections']
    def _apply(self,group):
        # make choosing _fast/_slow configurable (while testing)
        if flags['use_KDTree_routines'] in (True,'fast','always'):
            return self._apply_KDTree(group)
        else:
            return self._apply_distmat(group)
    def _apply_KDTree(self,group):
        """KDTree based selection is about 7x faster than distmat for typical problems.
        Limitations: always ignores periodicity
        """
        sel_atoms = self.sel._apply(group) ## group is wrong, should be universe (?!)
        sys_atoms_list = [a for a in (self._group_atoms-sel_atoms)]  # list needed for back-indexing
        sel_indices = numpy.array([a.number for a in sel_atoms],dtype=int)
        sys_indices = numpy.array([a.number for a in sys_atoms_list],dtype=int)
        sel_coor = Selection.coord[sel_indices]
        sys_coor = Selection.coord[sys_indices]
        from MDAnalysis.KDTree.NeighborSearch import CoordinateNeighborSearch
        # Can we optimize search by using the larger set for the tree?
        CNS = CoordinateNeighborSearch(sys_coor)  # cache the KDTree for this selection/frame?
        found_indices = CNS.search_list(sel_coor,self.cutoff)
        res_atoms = [sys_atoms_list[i] for i in found_indices] # make list numpy array and use fancy indexing?
        return set(res_atoms)
    def _apply_distmat(self,group):
        sel_atoms = self.sel._apply(group) ## group is wrong, should be universe (?!)
        sys_atoms_list = [a for a in (self._group_atoms-sel_atoms)]  # list needed for back-indexing
        sel_indices = numpy.array([a.number for a in sel_atoms],dtype=int)
        sys_indices = numpy.array([a.number for a in sys_atoms_list],dtype=int)
        sel_coor = Selection.coord[sel_indices]
        sys_coor = Selection.coord[sys_indices]
        if self.periodic:
            box = group.dimensions[:3]  # ignored with KDTree
        else:
            box = None
        import distances
        dist = distances.distance_array(sys_coor, sel_coor, box)
        res_atoms = [sys_atoms_list[i] for i in numpy.any(dist <= self.cutoff, axis=1).nonzero()[0]]  # make list numpy array and use fancy indexing?
        return set(res_atoms)
    def __repr__(self):
        return "<'AroundSelection' "+repr(self.cutoff)+" around "+repr(self.sel)+">"

class PointSelection(Selection):
    def __init__(self, x, y, z, cutoff, periodic=None):
        Selection.__init__(self)
        self.ref = numpy.array((float(x), float(y), float(z)))
        self.cutoff = float(cutoff)
        self.cutoffsq = float(cutoff)*float(cutoff)
        if periodic is None:
            self.periodic = flags['use_periodic_selections']
    def _apply(self,group):
        # make choosing _fast/_slow configurable (while testing)
        if flags['use_KDTree_routines'] in ('always',):
            return self._apply_KDTree(group)
        else:
            return self._apply_distmat(group)
    def _apply_KDTree(self, group):
        """Selection using KDTree but periodic = True not supported.
        (KDTree routine is ca 15% slower than the distance matrix one)
        """
        sys_indices = numpy.array([a.number for a in self._group_atoms_list])
        sys_coor = Selection.coord[sys_indices]
        if self.periodic:
            pass # or warn? -- no periodic functionality with KDTree search
        from MDAnalysis.KDTree.NeighborSearch import CoordinateNeighborSearch
        CNS = CoordinateNeighborSearch(sys_coor)  # cache the KDTree for this selection/frame?
        found_indices = CNS.search(self.ref,self.cutoff)
        res_atoms = [self._group_atoms_list[i] for i in found_indices]  # make list numpy array and use fancy indexing?
        return set(res_atoms)
    def _apply_distmat(self, group):
        """Selection that computes all distances."""
        sys_indices = numpy.array([a.number for a in self._group_atoms_list])
        sys_coor = Selection.coord[sys_indices]
        ref_coor = self.ref[numpy.newaxis,...]
        if self.periodic:
            box = group.dimensions[:3]
        else:
            box = None
        import distances
        dist = distances.distance_array(sys_coor, ref_coor, box)
        res_atoms = [self._group_atoms_list[i] for i in numpy.any(dist <= self.cutoff, axis=1).nonzero()[0]]   # make list numpy array and use fancy indexing?
        return set(res_atoms)
    def __repr__(self):
        return "<'PointSelection' "+repr(self.cutoff)+" Ang around "+repr(self.ref)+">"

class CompositeSelection(Selection):
    def __init__(self, name=None, type=None, resname=None, resid=None, segid=None):
        Selection.__init__(self)
        self.name = name
        self.type = type
        self.resname = resname
        self.resid = resid
        self.segid = segid
    def _apply(self, group):
        res = []
        for a in group.atoms:
            add = True
            if (self.name != None and a.name != self.name):
                add = False
            if (self.type != None and a.type != self.type):
                add = False
            if (self.resname != None and a.resname != self.resname):
                add = False
            if (self.resid != None and a.resid != self.resid):
                add = False
            if (self.segid != None and a.segid != self.segid):
                add = False
            if (add): res.append(a)
        return set(res)

class AtomSelection(Selection):
    def __init__(self, name, resid, segid):
        Selection.__init__(self)
        self.name = name
        self.resid = resid
        self.segid = segid
    def _apply(self, group):
        for a in group.atoms:
            if ((a.name == self.name) and (a.resid == self.resid) and (a.segid == self.segid)):
                return set([a])
        return set([])
    def __repr__(self):
        return "<'AtomSelection' "+repr(self.segid)+" "+repr(self.resid)+" "+repr(self.name)+" >"


class StringSelection(Selection):
    def __init__(self, field):
        Selection.__init__(self)
        self._field = field
    def _apply(self, group):
        # Look for a wildcard
        value = getattr(self, self._field)
        wc_pos = value.find('*')  # This returns -1, so if it's not in value then use the whole of value
        if wc_pos == -1: wc_pos = None
        return set([a for a in group.atoms if getattr(a, self._field)[:wc_pos] == value[:wc_pos]])
    def __repr__(self):
        return "<"+repr(self.__class__.__name__)+": "+repr(getattr(self, self._field))+">"

class AtomNameSelection(StringSelection):
    def __init__(self, name):
        StringSelection.__init__(self, "name")
        self.name = name

class AtomTypeSelection(StringSelection):
    def __init__(self, type):
        StringSelection.__init__(self, "type")
        self.type = type

class ResidueNameSelection(StringSelection):
    def __init__(self, resname):
        StringSelection.__init__(self, "resname")
        self.resname = resname

class SegmentNameSelection(StringSelection):
    def __init__(self, segid):
        StringSelection.__init__(self, "segid")
        self.segid = segid

class ByResSelection(Selection):
    def __init__(self, sel):
        Selection.__init__(self)
        self.sel = sel
    def _apply(self, group):
        res = self.sel._apply(group)
        unique_res = set([(a.resid, a.segid) for a in res])
        sel = []
        for atom in group.atoms:
            if (atom.resid, atom.segid) in unique_res:
                sel.append(atom)
        return set(sel)
    def __repr__(self):
        return "<'ByResSelection'>"

class ResidueIDSelection(Selection):
    def __init__(self, lower, upper):
        Selection.__init__(self)
        self.lower = lower
        self.upper = upper
    def _apply(self, group):
        if self.upper != None:
            return set([a for a in group.atoms if (self.lower <= a.resid <= self.upper)])
        else: return set([a for a in group.atoms if a.resid == self.lower])
    def __repr__(self):
        return "<'ResidueIDSelection' "+repr(self.lower)+":"+repr(self.upper)+" >"

class ByNumSelection(Selection):
    def __init__(self, lower, upper):
        Selection.__init__(self)
        self.lower = lower
        self.upper = upper
    def _apply(self, group):
        if self.upper != None:
            # In this case we'll use 1 indexing since that's what the user will be 
            # familiar with
            return set(group.atoms[self.lower-1:self.upper])
        else: return set(group.atoms[self.lower-1:self.lower])
    def __repr__(self):
        return "<'ByNumSelection' "+repr(self.lower)+":"+repr(self.upper)+" >"

class ProteinSelection(Selection):
    """A protein selection consists of all residues with  recognized residue names.
    Recognized residue names:
    * from the Charmm force field
         awk '/RESI/ {printf "'"'"%s"'"',",$2 }' top_all27_prot_lipid.rtf
    * manually added:
         HIS CHO EAM
    """
    prot_res = dict([(x,None) for x in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'HSD',
                                        'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
                                        'TRP', 'TYR', 'VAL', 'ALAD',
                                        'CHO', 'EAM']])
    def _apply(self, group):
        return set([a for a in group.atoms if a.resname in self.prot_res])
    def __repr__(self):
        return "<'ProteinSelection' >"

class NucleicSelection(Selection):
    """A nucleic selection consists of all residues with  recognized residue names.
    Recognized residue names:
    * from the Charmm force field
      awk '/RESI/ {printf "'"'"%s"'"',",$2 }' top_all27_prot_na.rtf
    """
    nucl_res = dict([(x,None) for x in ['ADE', 'URA', 'CYT', 'GUA', 'THY']])
    def _apply(self, group):
        return set([a for a in group.atoms if a.resname in self.nucl_res])
    def __repr__(self):
        return "<'NucleicSelection' >"

class BackboneSelection(ProteinSelection):
    """A BackboneSelection contains all atoms with name 'N', 'CA', 'C', 'O'.
  
    This excludes OT* on C-termini (which are included by, eg VMD's backbone selection).
    """
    bb_atoms = dict([(x,None) for x in ['N', 'CA', 'C', 'O']])
    def _apply(self, group):
       return set([a for a in group.atoms if (a.name in self.bb_atoms and a.resname in self.prot_res)])
    def __repr__(self):
        return "<'BackboneSelection' >"

class NucleicBackboneSelection(NucleicSelection):
    """A NucleicBackboneSelection contains all atoms with name "P", "C5'", C3'", "O3'", "O5'".
    """
    bb_atoms = dict([(x,None) for x in ['P', 'C5\'', 'C3\'', 'O3\'','O5\'']])
    def _apply(self, group):
        return set([a for a in group.atoms if (a.name in self.bb_atoms and a.resname in self.nucl_res)])
    def __repr__(self):
        return "<'NucleicBackboneSelection' >"

class BaseSelection(NucleicSelection):
    """A protein selection consists of all residues with  recognized residue names.
        Recognized residue names:
       * from the Charmm force field
       awk '/RESI/ {printf "'"'"%s"'"',",$2 }' top_all27_prot_lipid.rtf
    """
    base_atoms = dict([(x,None) for x in ['N9', 'N7', 'C8', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6',
                                          'O6','N2','N6',
                                          'O2','N4','O4','C5M']])
    def _apply(self, group):
        return set([a for a in group.atoms if (a.name in self.base_atoms and a.resname in self.nucl_res)])
    def __repr__(self):
        return "<'BaseSelection' >"

class NucleicSugarSelection(NucleicSelection):
    """A NucleicSugarSelection contains all atoms with name 'C1\'', 'C2\'','C3\'', 'C4\'', 'O2\'','O4\'','O3\''.
    """
    sug_atoms = dict([(x,None) for x in ['C1\'', 'C2\'','C3\'', 'C4\'', 'O2\'','O4\'','O3\'']])
    def _apply(self, group):
        return set([a for a in group.atoms if (a.name in self.sug_atoms and a.resname in self.nucl_res)])
    def __repr__(self):
        return "<'NucleicSugarSelection' >"

class CASelection(BackboneSelection):
    def _apply(self, group):
        return set([a for a in group.atoms if (a.name == "CA" and a.resname in self.prot_res)])
    def __repr__(self):
        return "<'CASelection' >"

class BondedSelection(Selection):
    def __init__(self, sel):
        Selection.__init__(self)
        self.sel = sel
    def _apply(self, group):
        res = self.sel._apply(group)
        # Find all the atoms bonded to each
        sel = []
        for atom in res:
            for b1, b2 in group._bonds:
                if atom.number == b1:
                    sel.append(group.atoms[b2])
                elif atom.number == b2:
                    sel.append(group.atoms[b1])
        return set(sel)
    def __repr__(self):
        return "<'BondedSelection' to "+ repr(self.sel)+" >"

class PropertySelection(Selection):
    """Some of the possible properties:
    x, y, z, radius, mass,
    """
    def __init__(self, prop, operator, value, abs=False):
        Selection.__init__(self)
        self.prop = prop
        self.operator = operator
        self.value = value
        self.abs = abs
    def _apply(self, group):
        # For efficiency, get a reference to the actual numpy position arrays
        if self.prop in ("x", "y", "z"):
            p = getattr(Selection.coord, '_'+self.prop)
            indices = numpy.array([a.number for a in group.atoms])
            if not self.abs:
                # XXX Hack for difference in numpy.nonzero between version < 1. and version > 1
                res = numpy.nonzero(self.operator(p[indices], self.value))
            else:
                res = numpy.nonzero(self.operator(numpy.abs(p[indices]), self.value))
            if type(res) == tuple: res = res[0]
            result_set = [group.atoms[i] for i in res]
        elif self.prop == "mass":
            result_set = [a for a in group.atoms if self.operator(a.mass,self.value)]
        elif self.prop == "charge":
            result_set = [a for a in group.atoms if self.operator(a.charge,self.value)]
        return set(result_set)
    def __repr__(self):
        if self.abs: abs_str = " abs "
        else: abs_str = ""
        return "<'PropertySelection' "+abs_str+repr(self.prop)+" "+repr(self.operator.__name__)+" "+repr(self.value)+">"

class ParseError(Exception):
    pass

class SelectionParser:
    """A small parser for selection expressions.  Demonstration of
    recursive descent parsing using Precedence climbing (see
    http://www.engr.mun.ca/~theo/Misc/exp_parsing.htm).  Transforms
    expressions into nested Selection tree.

    For reference, the grammar that we parse is :

    E(xpression)--> Exp(0)
    Exp(p) -->      P {B Exp(q)}
    P -->           U Exp(q) | "(" E ")" | v
    B(inary) -->    "and" | "or"
    U(nary) -->     "not"
    T(erms) -->     segid [value]
                    | resname [value]
                    | resid [value]
                    | name [value]
                    | type [value]
   """

    #Here are the symbolic tokens that we'll process:
    ALL = 'all'
    NOT = 'not'
    AND = 'and'
    OR = 'or'
    AROUND = 'around'
    POINT = 'point'
    BYRES = 'byres'
    BONDED = 'bonded'
    BYNUM = 'bynum'
    PROP = 'prop'
    ATOM = 'atom'
    LPAREN = '('
    RPAREN = ')'
    SEGID = 'segid'
    RESID = 'resid'
    RESNAME = 'resname'
    NAME = 'name'
    TYPE = 'type'
    PROTEIN = 'protein'
    NUCLEIC = 'nucleic'
    BB = 'backbone'
    NBB = 'nucleicbackbone'
    BASE = 'nucleicbase'
    SUGAR = 'nucleicsugar'
    EOF = 'EOF'
    GT = '>'
    LT = '<'
    GE = '>='
    LE = '<='
    EQ = '=='
    NE = '!='

    classdict = dict([(ALL, AllSelection), (NOT, NotSelection), (AND, AndSelection), (OR, OrSelection),
                      (SEGID, SegmentNameSelection), (RESID, ResidueIDSelection),
                      (RESNAME, ResidueNameSelection), (NAME, AtomNameSelection),
                      (TYPE, AtomTypeSelection), (BYRES, ByResSelection),
                      (BYNUM, ByNumSelection), (PROP, PropertySelection),
                      (AROUND, AroundSelection), (POINT, PointSelection),
                      (NUCLEIC, NucleicSelection), (PROTEIN, ProteinSelection), 
                      (BB, BackboneSelection), (NBB, NucleicBackboneSelection),
		      (BASE, BaseSelection), (SUGAR, NucleicSugarSelection),
                      #(BONDED, BondedSelection), not supported yet, need a better way to walk the bond lists
                      (ATOM, AtomSelection)])
    associativity = dict([(AND, "left"), (OR, "left")])
    precedence = dict([(AROUND, 1), (POINT, 1), (BYRES, 1), (BONDED, 1), (AND, 3), (OR, 3), (NOT,5 )])

    # Borg pattern: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66531
    _shared_state = {}
    def __new__(cls, *p, **k):
        self = object.__new__(cls, *p, **k)
        self.__dict__ = cls._shared_state
        return self

    def __peek_token(self):
        """Looks at the next token in our token stream."""
        return self.tokens[0]

    def __consume_token(self):
        """Pops off the next token in our token stream."""
        next_token = self.tokens[0]
        del self.tokens[0]
        return next_token

    def __error(self, token):
        """Stops parsing and reports and error."""
        raise ParseError("Parsing error- '"+self.selectstr+"'\n"+repr(token)+" expected")

    def __expect(self, token):
        if self.__peek_token() == token:
            self.__consume_token()
        else:
            self.__error(token)

    def parse(self, selectstr):
        self.selectstr = selectstr
        self.tokens = selectstr.split()+[self.EOF]
        parsetree = self.__parse_expression(0)
        self.__expect(self.EOF)
        return parsetree

    def __parse_expression(self, p):
        exp1 = self.__parse_subexp()
        while (self.__peek_token() in (self.AND, self.OR) and self.precedence[self.__peek_token()] >= p): # binary operators
            op = self.__consume_token()
            if self.associativity[op] == "right": q = self.precedence[op]
            else: q = 1 + self.precedence[op]
            exp2 = self.__parse_expression(q)
            exp1 = self.classdict[op](exp1, exp2)
        return exp1

    def __parse_subexp(self):
        op = self.__consume_token()
        if op in (self.NOT, self.BYRES): # unary operators
            exp = self.__parse_expression(self.precedence[op])
            return self.classdict[op](exp)
        elif op in (self.AROUND):
            dist = self.__consume_token()
            exp = self.__parse_expression(self.precedence[op])
            return self.classdict[op](exp, float(dist))
        elif op in (self.POINT):
            dist = self.__consume_token()
            x = self.__consume_token()
            y = self.__consume_token()
            z = self.__consume_token()
            return self.classdict[op](float(dist), float(x), float(y), float(z))
        elif op == self.BONDED:
            exp = self.__parse_expression(self.precedence[op])
            return self.classdict[op](exp)
        elif op == self.LPAREN:
            exp = self.__parse_expression(0)
            self.__expect(self.RPAREN)
            return exp
        elif op in (self.SEGID, self.RESNAME, self.NAME, self.TYPE):
            data = self.__consume_token()
            if data in (self.LPAREN, self.RPAREN, self.AND, self.OR, self.NOT, self.SEGID, self.RESID, self.RESNAME, self.NAME, self.TYPE):
                self.__error("Identifier")
            return self.classdict[op](data)
        elif op == self.PROTEIN:
            return self.classdict[op]()
	elif op == self.NUCLEIC:
	    return self.classdict[op]()
        elif op == self.ALL:
            return self.classdict[op]()
        elif op == self.BB:
            return self.classdict[op]()
        elif op == self.NBB:
            return self.classdict[op]()
	elif op == self.BASE:
	    return self.classdict[op]()
	elif op == self.SUGAR:
	    return self.classdict[op]()
        elif op == self.RESID:
            data = self.__consume_token()
            try:
                lower = int(data)
                upper = None
            except ValueError:
                import re
                selrange=re.match("(\d+)[:-](\d+)",data) # check if in appropriate format 'lower:upper' or 'lower-upper'
                if not selrange: self.__error(self.RESID)
                lower, upper = map(int, selrange.groups())
            return self.classdict[op](lower,upper)
        elif op == self.BYNUM:
            data = self.__consume_token()
            try:
                lower = int(data)
                upper = None
            except ValueError:
                import re
                selrange=re.match("(\d+)[:-](\d+)",data) # in the appropriate format 'lower:upper'
                if not selrange: self.__error(self.BYNUM)
                lower, upper = map(int, selrange.groups())
            return self.classdict[op](lower,upper)
        elif op == self.PROP:
            prop = self.__consume_token()
            if prop == "abs": 
                abs = True
                prop = self.__consume_token()
            else: abs = False
            oper = self.__consume_token()
            value = float(self.__consume_token())
            ops = dict([(self.GT, numpy.greater), (self.LT, numpy.less), 
                        (self.GE, numpy.greater_equal), (self.LE, numpy.less_equal),
                        (self.EQ, numpy.equal), (self.NE, numpy.not_equal)])
            if oper in ops.keys():
                return self.classdict[op](prop, ops[oper], value, abs)
        elif op == self.ATOM:
            segid = self.__consume_token()
            resid = int(self.__consume_token())
            name = self.__consume_token()
            return self.classdict[op](name, resid, segid)
        else:
            self.__error(op)

# The module level instance
Parser = SelectionParser()