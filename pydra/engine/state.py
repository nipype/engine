from collections import OrderedDict
import itertools
from copy import deepcopy
import pdb

from . import auxiliary as aux


class State(object):
    def __init__(self, node):
        self.node = node
        self._splitter = node.splitter
        self._other_splitters = node._other_splitters
        self.node_name = node.name
        self._inner_splitter = []
        self._inner_combiner = []
        self.comb_inp_to_remove = []

        self.state_inputs = node.state_inputs
        if hasattr(self.node, "interface"):
            self._inner_inputs_names = ["{}.{}".format(self.node_name, inp) for inp in self.node.inner_inputs_names]
            # adding inner splitters from other nodes
            self._inner_inputs_names = self._inner_inputs_names + self.node.wf_inner_splitters
            if self._splitter and self._inner_inputs_names:
                self._inner_splitter_separation(combiner=node.combiner)
        if not hasattr(self, "_splitter_wo_inner"):
            self._splitter_wo_inner = self._splitter
            self._combiner_wo_inner = node.combiner

        # changing splitter (as in rpn), so I can read from left to right
        # e.g. if splitter=('d', ['e', 'r']), _splitter_rpn=['d', 'e', 'r', '*', '.']
        self._splitter_rpn = aux.splitter2rpn(self._splitter, other_splitters=self._other_splitters)
        self._input_names_splitter = [i for i in self._splitter_rpn if i not in ["*", "."]]

        self._splitter_rpn_wo_inner = aux.splitter2rpn(self._splitter_wo_inner, other_splitters=self._other_splitters)

        # TODO: should try to change the order and move it higher
        if node.combiner:
            self.combiner = node.combiner
        else:
            self._combiner = node.combiner

        for spl in self._inner_splitter:
            if spl not in self.node.wf_inner_splitters:
                self.node.wf_inner_splitters.append(spl)

        self._inner_splitter_comb = list(set(self._inner_splitter) - set(self._inner_combiner))

    def prepare_state_input(self):
        """prepare all inputs, should be called once all input is available"""

        # not all input field have to be use in the splitter, can be an extra scalar
        self._input_names = list(self.state_inputs.keys())

        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self._axis_for_input, self._ndim = aux.splitting_axis(self.state_inputs, self._splitter_rpn_wo_inner)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        # TODO: do I need it?
        self._input_for_axis, self._shape = aux.converting_axis2input(
            state_inputs=self.state_inputs, axis_for_input=self._axis_for_input,
            ndim=self._ndim)

        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self.all_elements = [range(i) for i in self._shape]
        self.index_generator = itertools.product(*self.all_elements)

        if self.combiner:
            self._prepare_axis_inputs_combine()


    def __getitem__(self, ind):
        if type(ind) is int:
            ind = (ind, )
        return self.state_values(ind)

    # not used?
    #@property
    #def splitter(self):
    #    return self._splitter

    @property
    def combiner(self):
        return self._combiner

    @combiner.setter
    def combiner(self, combiner):
        # if self._combiner:
        #     raise Exception("combiner is already set")
        # else:
        self._combiner = combiner
        for el in self._combiner:
            if el not in self._splitter_rpn:
                raise Exception("element {} of combiner is not found in the splitter {}".format(
                    el, self._splitter))

        #self._combiner_wo_inner = list(set(self._combiner) - set(self._inner_splitter))
        self._prepare_combine()


    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape


    def _prepare_combine(self):
        # input and axes that will be removed when combiner applied
        self.comb_axes_to_remove = []
        # temporary axis_for_input (without knowing specific input)
        # need to know which inputs are bound together
        axis_for_input_tmp, ndim_tmp = aux.matching_input_from_splitter(self._splitter_rpn)
        input_for_axis_tmp = aux.converting_axis2input(axis_for_input=axis_for_input_tmp,
                                                       ndim=ndim_tmp)
        for comb_el in self._combiner_wo_inner:
            self.comb_inp_to_remove.append(comb_el)
            axes = axis_for_input_tmp[comb_el]
            for ax in axes:
                if ax in self.comb_axes_to_remove:
                    raise Exception("can't combine by {}, axis {} already removed".format(
                        comb_el, ax
                    ))
            self.comb_axes_to_remove += axes

        # sorting and removing repeated values
        self.comb_axes_to_remove = list(set(self.comb_axes_to_remove))
        self.comb_axes_to_remove.sort()

        # (original axis, new axis) for axes that remain after combining
        self.partial_comb_input_rem_axes = {}
        # (original axis of the input, axis from the combiner space) for axes that are combined
        self.partial_comb_input_comb_axes = {}
        for ax in self.comb_axes_to_remove:
            inputs = deepcopy(input_for_axis_tmp[ax])
            for other_inp in set(inputs) - set(self._combiner_wo_inner):
                self.comb_inp_to_remove.append(other_inp)
                # saving remaining axes and position in a final splitter
                remaining_axes = set(axis_for_input_tmp[other_inp])\
                                 - set(self.comb_axes_to_remove)
                if remaining_axes:
                    self.partial_comb_input_rem_axes[other_inp] = []
                    for rem_ax in remaining_axes:
                        axes_shift = (rem_ax, rem_ax - len([i for i in self.comb_axes_to_remove
                                                            if i < rem_ax]))
                        self.partial_comb_input_rem_axes[other_inp].append(axes_shift)

                combined_axes = list(set(axis_for_input_tmp[other_inp]) - set(remaining_axes))
                combined_axes.sort()
                if combined_axes:
                    self.partial_comb_input_comb_axes[other_inp] = []
                    for ind_ax, comb_ax in enumerate(combined_axes):
                        axes_shift = (ind_ax, self.comb_axes_to_remove.index(comb_ax))
                        self.partial_comb_input_comb_axes[other_inp].append(axes_shift)

        self._prepare_splitter_combine()


    # TODO: should I call it from splitter?
    def _prepare_splitter_combine(self):
        self._splitter_rpn_comb = aux.remove_inp_from_splitter_rpn(self._splitter_rpn_wo_inner,
                                                                   self.comb_inp_to_remove)
        self.splitter_comb = aux.rpn2splitter(self._splitter_rpn_comb)


    def _prepare_axis_inputs_combine(self):
        # todo: do i need it?
        self._state_inputs_comb = self.state_inputs.copy()
        for inp in self.comb_inp_to_remove:
            self._state_inputs_comb.pop(inp)
        self._axis_for_input_comb, self._ndim_comb = aux.splitting_axis(self._state_inputs_comb,
                                                                         self._splitter_rpn_comb)
        self._input_for_axis_comb, self._shape_comb = aux.converting_axis2input(
            state_inputs=self._state_inputs_comb, axis_for_input=self._axis_for_input_comb,
            ndim=self._ndim_comb)


    def _inner_splitter_separation(self, combiner):
        """
        checking if splitter is ok, allowing for inner splitter,
        assuming that inner splitter can be only in the most outer layer,
        and that it has to be also in the combiner
        """
        if type(self._splitter) is str:
            if self._splitter in self._inner_inputs_names:
                self._inner_splitter.append(self._splitter)
                self._splitter_wo_inner = None
        elif type(self._splitter) is tuple:
            if all([x in self._inner_inputs_names for x in self._splitter]):
                # TODO: still they might not gave the same shapes...
                self._inner_splitter += list(self._splitter)
                self._splitter_wo_inner = None
            elif any([x in self._inner_inputs_names for x in self._splitter]):
                    raise Exception("the scalar splitter {} is not correct, either both or neither "
                                    "od the elements should be inner inputs".format(self._splitter))
        elif type(self._splitter) is list:
            if all([x in self._inner_inputs_names for x in self._splitter]):
                # TODO: should i allow it?
                raise Exception("the outer splitter {} is not correct, both elements "
                                "can't be from inner inputs".format(self._splitter))
            # checking if one of the element is an inner input
            else:
                for (i, spl) in enumerate(self._splitter):
                    if spl in self._inner_inputs_names:
                        self._inner_splitter.append(spl)
                        self._splitter_wo_inner = self._splitter[(i+1)%2]

        self._inner_combiner = [comb for comb in combiner if comb in self._inner_splitter]
        self._combiner_wo_inner = list(set(combiner) - set(self._inner_combiner))
        # pdb.set_trace()
        # pass

        # checking if there are no more inner inputs or not known inputs in the splitter
        # _input_names_splitter_copy = copy(self._input_names_splitter)
        # [_input_names_splitter_copy.remove(x) for x in self._inner_splitter]
        # for inp in _input_names_splitter_copy:
        #     if inp in self._input_names:
        #         pass
        #     elif inp in self._inner_inputs_names:
        #         raise Exception("inner input {} can be only once in the splitter in the "
        #                         "most outer part".format(inp))
        #     else:
        #         raise Exception("input name {} from splitter not known".format(inp))


    def state_values(self, ind, value=True):
        """returns state input as a dictionary (input name, value)"""
        if len(ind) > self._ndim:
            raise IndexError("too many indices")

        for ii, index in enumerate(ind):
            if index > self._shape[ii] - 1:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(
                    index, ii, self._shape[ii]))

        state_dict = {}
        for input, ax in self._axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1] + 1)
            # taking the indexes for the axes
            ind_inp = tuple(ind[sl_ax])  #used to be list
            if value:
                state_dict[input] = self.state_inputs[input][ind_inp]
            else: # using index instead of value
                ind_inp_str = "x".join([str(el) for el in ind_inp])
                state_dict[input] = ind_inp_str
        # adding values from input that are not used in the splitter
        for input in set(self._input_names) - set(self._input_names_splitter):
            if value:
                state_dict[input] = self.state_inputs[input]
            else:
                state_dict[input] = None
        # in py3.7 we can skip OrderedDict
        # returning a named tuple?
        return OrderedDict(sorted(state_dict.items(), key=lambda t: t[0]))


    def state_ind(self, ind):
        """state_values, but returning indices instead of values"""
        return self.state_values(ind, value=False)