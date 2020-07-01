'''
Calculate the activity of the CO2 reduction reaction given the binding energy
of CO.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import numpy as np
from scipy import stats


X_LHS = [-1.3885017421602783, -1.3327526132404182, -1.200348432055749, -1.012195121951219, -0.6428571428571423, -0.4198606271776999, -0.1829268292682924]
Y_LHS = [-1.5009528942961072, -1.17408127514582, -0.6878164282826713, -0.20617167497641908, 0.9234411996842962, 1.899435963578263, 2.708529847729416]

X_RHS = [-0.17595818815331032, -0.015679442508711006, 0.2560975609756104, 0.4233449477351918, 0.6881533101045294]
Y_RHS = [2.542206479681214, 1.202956859876414, -1.4714997978709086, -3.142818642077504, -5.650951931776618]

M_LHS, B_LHS, _, _, _ = stats.linregress(X_LHS, Y_LHS)
M_RHS, B_RHS, _, _, _ = stats.linregress(X_RHS, Y_RHS)


def calc_co2rr_activity(eCO):
    '''
    Calculates the activity of the CO2 reduction reaction given the binding
    energy of CO. We do this with the 211 volcano relationship in Figure 4b of
    https://www.nature.com/articles/ncomms15438.

    Arg:
        eCO     Binding energy of CO (dE, not dG. We assume dE is 0.5 eV less
                than dG). We assume units of eV.
    Returns:
        activity    Reaction rate of CO2 reduction in mA/cm**2
    '''
    dG = eCO + 0.5
    if eCO < -0.67:
        ln_activity = _calc_activity_lhs(dG)
    else:
        ln_activity = _calc_activity_rhs(dG)
    activity = np.exp(ln_activity)
    return activity


def _calc_activity_lhs(dG):
    activity = M_LHS * dG + B_LHS
    return activity


def _calc_activity_rhs(dG):
    activity = M_RHS * dG + B_RHS
    return activity
