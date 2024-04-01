import numpy as np
from scipy import stats
from statistics import mode as mode_function

def calculate_statistics(dataset):
    #mean
    mean = np.mean(dataset)
    
    #mode
    mode_value = mode_function(dataset)
    
    #median
    medain = np.median(dataset)
    
    #variance 
    variance = np.var(dataset)
    
    #standerd deviation
    std_dev = np.std(dataset)
    
    #quartiles
    
    quartiles = np.percentile(dataset, [25,50,75])
    
    #interqurtile range
    
    iqr = quartiles[2] - quartiles[0] 
    
    return{
        "Mean" : mean,
        "Mode" : mode_value,
        "Median" : medain,
        "variance" : variance,
        "std" : std_dev,
        "Quartiles" : quartiles,
        "Interqurtile" : iqr
    }
    
dataset = [27, 14, 92, 37, 64, 27, 14, 50, 64, 92,
           37, 50, 27, 92, 14, 50, 37, 64, 92, 14,
           27, 64, 50, 37, 92, 14, 27, 50, 64, 37,
           14, 92, 27, 64, 50, 14, 37, 92, 27, 64,
           50, 92, 37, 14, 64, 50, 27, 14, 37, 92]

#calculate statastics

statistics = calculate_statistics(dataset)

#display Stataistics

for key, value in statistics.items():
    print(f"{key}:{value}")