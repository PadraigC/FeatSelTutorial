from CFS import merit_calculation
import pandas as pd
import numpy as np

def CFS_FS(X,y):
    
    '''
    This function performs a forward search for CFS
    Inputs:
    X - training data
    y - labels
    
    Outputs:
    merit_score_sel - The merit value assigned to the selected feature subsets in the order they were added
    sel_comb - The selected feature combination
    '''

    # initialise variables
    var_no = 1
    sel_comb = []
    merit_score_change = 1
    merit_score_prev = 0
    merit_score_sel = pd.DataFrame()
    enum = 0

    m,n = X.shape

    for  i in range(0,n-1):
    
        # Create a consecutive list with all the variables
        var_list = list(range(0,n))
        combs = []
        j = 0
        
        # Find the unique  combinations of variables
        if(var_no ==1):
            combs = var_list
        elif (var_no == 2):
            var_list.remove(sel_comb)
            for i in var_list:
                combs.insert(j, tuple([sel_comb,i]))
                j=j+1
        else:
            for i in sel_comb:
                var_list.remove(i)
            for i in var_list:
                combs.insert(j, sel_comb + (i,)) 
                j=j+1
            
        # Iterate through the possible feature subsets and find merit scores
        merit_score = []
        for i in range(0,len(combs)):
            X_input = X[:,combs[i]]
            if (var_no == 1):
                X_input = np.atleast_2d(X_input).T
            MS = merit_calculation(X_input, y)
            merit_score.append(MS)

        # Calculate the change in the merit score, once the score stops improving, stop the search
        merit_score_change = max(merit_score) - merit_score_prev
        if(merit_score_change <= 0):
            break
        else:
            sel_comb = combs[np.argmax(merit_score)]
            merit_score_prev = max(merit_score)
            var_no = var_no + 1

            merit_score_sel.insert(enum, enum,[ merit_score_prev])
            enum = enum+1
        
    return merit_score_sel, sel_comb