import pathlib
import pandas as pd


def cost_degradation_breakdown(df, h_analysis_path, p_analysis_path):
    degradations = df['degradation'].unique()
    bounds = df['bound'].unique()
    with open(h_analysis_path, 'w+') as h_file, open(p_analysis_path, 'w+') as p_file:
        bounds_str = ','.join(['d/b'] + [str(i) for i in bounds])
        h_file.write(bounds_str + '\n')
        p_file.write(bounds_str + '\n')
        for degradation in degradations:
            h_cline_str = str(degradation)+','
            p_cline_str = str(degradation)+','
            for bound in bounds:
                cdf = df[(df['bound'] == bound) & (df['degradation'] == degradation)]
                h_expanded_avg = round(cdf['h_expanded'].mean(), 2)
                h_expanded_std = round(cdf['h_expanded'].std(), 2)
                h_expanded_med = cdf['h_expanded'].median()
                p_expanded_avg = round(cdf['p_expanded'].mean(), 2)
                p_expanded_std = round(cdf['p_expanded'].std(), 2)
                p_expanded_med = cdf['p_expanded'].median()
                h_cline_str += f'{h_expanded_avg} ({h_expanded_std})|{h_expanded_med},'
                p_cline_str += f'{p_expanded_avg} ({p_expanded_std})|{p_expanded_med},'
            h_cline_str = h_cline_str[:-1] + '\n'
            p_cline_str = p_cline_str[:-1] + '\n'
            h_file.write(h_cline_str)
            p_file.write(p_cline_str)


def find_and_remove_nosolutions(df):
    error_rows = df.loc[(df['h_cost'] == -2) | (df['p_cost'] == -2)]
    if len(error_rows) > 0:
        error_rows_desc = error_rows[['instance_id', 'degradation', 'bound']].to_records(index=False)
        raise Exception(f"Found -2 (ERROR) in following searches: {error_rows_desc}")
    return df.drop(df[(df['h_cost'] == -1) | (df['p_cost'] == -1)].index)


def main():
    num_of_pancakes = 14
    parent_dir = pathlib.Path.cwd().parent
    results_path = parent_dir.joinpath('files').joinpath(f'pancakes_results_{num_of_pancakes}.csv')
    df = pd.read_csv(results_path)
    df = find_and_remove_nosolutions(df)
    plots_path = parent_dir.joinpath('plots')
    cost_degradation_breakdown(df,
                               plots_path.joinpath(f'pancakes_heu_bd_analysis_{num_of_pancakes}.csv'),
                               plots_path.joinpath(f'pancakes_pts_bd_analysis_{num_of_pancakes}.csv'))


if __name__ == '__main__':
    main()
