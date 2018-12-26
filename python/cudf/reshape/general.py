# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from cudf.dataframe import Series
from cudf.dataframe import Buffer
from cudf.dataframe import DataFrame
from cudf.utils import cudautils
from cudf.dataframe.categorical import CategoricalColumn

from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm

def melt(frame, id_vars=None, value_vars=None, var_name='variable',
         value_name='value'):
    """Unpivots a DataFrame from wide format to long format,
    optionally leaving identifier variables set.
    
    Parameters
    ----------
    frame : DataFrame
    id_vars : tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
        default: None
    value_vars : tuple, list, or ndarray, optional
        Column(s) to unpivot.
        default: all columns that are not set as `id_vars`.
    var_name : scalar
        Name to use for the `variable` column.
        default: frame.columns.name or 'variable'
    value_name : str
        Name to use for the `value` column.
        default: 'value'

    Returns
    -------
    molten : DataFrame

    Difference from pandas:
     * Does not support 'col_level' because cuDF does not have multi-index

    TODO: Examples
    """

    # Arg cleaning
    import types
    import collections
    # id_vars
    if id_vars is not None:
        if not isinstance(id_vars, collections.abc.Sequence):
            id_vars = [id_vars]
        id_vars = list(id_vars)
        missing = set(id_vars) - set(frame.columns)
        if not len(missing) == 0:
            raise KeyError(
                "The following 'id_vars' are not present"
                " in the DataFrame: {missing}"
                "".format(missing=list(missing)))
    else:
        id_vars = []

    # value_vars
    if value_vars is not None:
        if not isinstance(value_vars, collections.abc.Sequence):
            value_vars = [value_vars]
        value_vars = list(value_vars)
        missing = set(value_vars) - set(frame.columns)
        if not len(missing) == 0:
            raise KeyError(
                "The following 'value_vars' are not present"
                " in the DataFrame: {missing}"
                "".format(missing=list(missing)))
    else:
        # then all remaining columns in frame
        value_vars = frame.columns.drop(id_vars)
        value_vars = list(value_vars)
    
    if len(value_vars) != 0:
        dtypes = [ frame[var].dtype for var in value_vars ]
        dtype = dtypes[0]
        if pd.api.types.is_categorical_dtype(dtype):
            raise NotImplementedError('Categorical columns are not yet '
                                        'supported for function')
        if any(t != dtype for t in dtypes):
            raise ValueError('all columns must have the same dtype')

    # overlap
    overlap = set(id_vars).intersection(set(value_vars))
    if not len(overlap) == 0:
        raise KeyError(
            "'value_vars' and 'id_vars' cannot have overlap."
            " The following 'value_vars' are ALSO present"
            " in 'id_vars': {overlap}"
            "".format(overlap=list(overlap)))

    N = len(frame)
    K = len(value_vars)

    id_cols_ptr = [
        frame[col_name]._column.cffi_view
        for col_name in id_vars
    ]
    value_cols_ptr = [
        frame[col_name]._column.cffi_view
        for col_name in value_vars
    ]

    new_nrow = N * K
    dtypes = []
    for col in id_vars:
        dtypes.append(frame[col].dtype)
    dtypes.append(np.int8) # for the categorical
    dtypes.append(frame[value_vars[0]].dtype) # for the value column

    new_col_series = [
        Series.from_masked_array(
            data=Buffer(rmm.device_array(shape=new_nrow, dtype=dt)),
            mask=cudautils.make_mask(size=new_nrow),
        )
        for dt in dtypes]
    new_col_series_ptr = [ ser._column.cffi_view for ser in new_col_series ]

    libgdf.gdf_melt(
        id_cols_ptr,
        len(id_cols_ptr),
        value_cols_ptr,
        len(value_cols_ptr),
        new_col_series_ptr
    )

    new_mdata = collections.OrderedDict()
    # add id series to dict
    for col_name in id_vars:
        new_mdata[col_name] = new_col_series.pop(0)
    # add variable series to dict
    temp = new_col_series.pop(0)
    new_mdata[var_name] = Series(CategoricalColumn(
        categories=tuple(value_vars), data=temp._column.data, ordered=False))
    new_mdata[value_name] = new_col_series.pop(0)
    
    return DataFrame(new_mdata)
