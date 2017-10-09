# -*- coding: utf-8 -*-

"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation

Input : --run json
        Note: This json structure may involve different
                fields for different runs
Output: json
"""

import argparse
import json
import numpy as np
import sys
import regression as reg


def local_1(args, computation_phase):
    """
    Args:
        args (dictionary) : {"input":
                                {"covariates": ,
                                 "dependents": ,
                                 "lambda":
                                },
                            "cache": {}
                            }

        computation_phase () : ???

    Returns:
        computation_output : json

    Description:
        Step 1 : Generate the local beta_vector
        Step 2 : Generate the local fit statistics
                    r^2 : 1 - (SSE/SST)
                    t   : beta/std.err(beta)
                    p   : two-tailed p-value
        Step 3 : Generate the mean_y_local and count_local

    Example:
        python local.py --run '{"input":
                                {"covariates":
                                    [[2,3],[3,4],[7,8],[7,5],[9,8]],
                                "dependents":
                                    [6,7,8,5,6],
                                "lambda":0
                                },
                            "cache":{}
                            }'
    """
    input_list = args['input']
    X = input_list['covariates']
    y = input_list['dependents']
    lamb = input_list['lambda']
    biased_X = np.insert(X, 0, 1, axis=1)

    beta_vector = reg.one_shot_regression(X, y, lamb)

    r_squared = reg.r_square(biased_X, y, beta_vector)
    ts_beta = reg.t_value(biased_X, y, beta_vector)
    dof = len(y) - len(beta_vector)
    ps_beta = reg.t_to_p(dof, ts_beta)

    computation_output = json.dumps(
            {'output': {'beta_vector_local': beta_vector.tolist(),
                        'r_2_local': r_squared,
                        'ts_local': ts_beta.tolist(),
                        'ps_local': ps_beta,
                        'mean_y_local': np.mean(y),
                        'count_local': len(y),
                        'computation_phase': computation_phase
                        },
             'cache': {'covariates': X,
                       'dependents': y,
                       'lambda': lamb
                       }
             },
            sort_keys=True,
            indent=4,
            separators=(',', ': ')
            )

    return computation_output


def local_2(args, computation_phase):
    """
    Args:
        args (dictionary)       :
        computation_phase ()    :

    Returns:
        computation_output      : json dump

    Algorithm:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    cache_list = args['cache']
    input_list = args['input']

    X = cache_list['covariates']
    y = cache_list['dependents']
    biased_X = np.insert(X, 0, 1, axis=1)

    avg_beta_vector = input_list['avg_beta_vector']
    mean_y_global = input_list['mean_y_global']

    SSE_local = reg.sum_squared_error(biased_X, y, avg_beta_vector)
    SST_local = np.sum(np.square(y - mean_y_global))
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = json.dumps({'output':
                                    {'SSE_local': SSE_local,
                                     'SST_local': SST_local,
                                     'varX_matrix_local': varX_matrix_local,
                                     'computation_phase': computation_phase}},
                                    sort_keys=True,
                                    indent=4,
                                    separators=(',', ': '))
    return computation_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='''read in coinstac args for
                            local computation'''
                            )
    parser.add_argument('--run', type=json.loads,  help='grab coinstac args')
    args = parser.parse_args()
    input_list = args.run['input']

    if 'computation_phase' not in input_list.keys():
        computation_phase = 'local_1'
        computation_output = local_1(args.run, computation_phase)
        sys.stdout.write(computation_output)
    elif input_list['computation_phase'] == 'remote_1':
        computation_phase = 'local_2'
        computation_output = local_2(args.run, computation_phase)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Invalid value for computation_phase')
