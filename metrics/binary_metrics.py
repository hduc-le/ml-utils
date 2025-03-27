import pandas as _pd 
import numpy as _np

def assign_tag(inp:_pd.DataFrame, score_col:str='prob', tag_col:str='tag', n=10):
    inp = inp.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    inp[tag_col] = ((inp.index) // (inp.shape[0] // n)) + 1
    inp[tag_col] = _np.minimum(inp[tag_col], n).astype(_np.int32)
    return inp
#
def get_pr_rc_by_tag(inp: _pd.DataFrame, cname_target:str='flag_bad', tag_col:str='tag'):
    result = inp.groupby(tag_col).agg({cname_target: ['count', 'sum']})
    result = flatten_multilevel_columns(result).sort_index().reset_index()
    result = result.rename(columns={f'{cname_target}_sum': 'n_positives', f'{cname_target}_count': 'n_samples'})
    # Calculate cumulative sum
    result['n_samples_cumsum'] = result['n_samples'].cumsum()
    result['n_positives_cumsum'] = result['n_positives'].cumsum()

    # Calculate percentage
    result['percentage_samples'] = result['n_samples'].apply(lambda x: x / result['n_samples'].sum() * 100)
    result['percentage_positives'] = result['n_positives'].apply(lambda x: x / result['n_positives'].sum() * 100)
    result['Precision (%)'] = result['n_positives'] / result['n_samples'] * 100
    result['Precision_acc (%)'] = result['n_positives'].cumsum() / result['n_samples'].cumsum() * 100
    result['Recall (%)'] = result['n_positives'] / result['n_positives'].sum() * 100
    result['Recall_acc (%)'] = result['n_positives'].cumsum() / result['n_positives'].sum() * 100
    return result
#
def flatten_multilevel_columns(inp):
    inp.columns = ['_'.join(c).strip() if c[1] != '' else c[0] for c in inp.columns.values]
    return inp