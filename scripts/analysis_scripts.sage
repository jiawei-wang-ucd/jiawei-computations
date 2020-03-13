import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats; from scipy.stats import sem, t, kurtosis, kurtosistest, skew
from scipy.stats.mstats import gmean

np.random.seed(9001)

def generate_plots_interval(x_values, y_values_upper, y_values_lower, y_values_mean, log_plot = False, **kwags):
    if log_plot:
        y_values_lower = [log(v) for v in y_values_lower]
        y_values_upper = [log(v) for v in y_values_upper]
        y_values_mean = [log(v) for v in y_values_mean]



def shifted_geometric_mean(l, s):
    """
    Compute the shifted geometric mean of a list l with shift s.
    """
    if s<0:
        raise ValueError("The shift s should not be negative.")
    return gmean([v+s for v in l])-s

def bootstrap_CI(l, N=10000, alpha=0.95):
    """
    Compute the confidence interval of the sample mean using bootstrap.
    """
    mean_list = [mean(np.random.choice(l,len(l),replace=True)) for i in xrange(N)]
    return np.percentile(mean_list,(1-alpha)/2*100), np.percentile(mean_list,(1+alpha)/2*100)

def t_distribution_CI(l, alpha=0.95):
    """
    Compute the confidence interval of the sample mean using t distribution.
    """
    n = len(l)
    m = mean(l)
    std_err = sem(l)
    h = std_err * t.ppf((1 + alpha) / 2, n - 1)
    return m-h, m+h

def sample_skew(l):
    return skew(l)

def sample_kurtosis(l):
    return kurtosis(l)

def kurtosis_test_p_value(l):
    """
    Hypothesis test to compute the p value under the null hypothesis that the samples are drawn from a normal distribution.
    """
    if len(l)<20:
        raise ValueError("Not enough data")
    res = kurtosistest(l)
    return res.pvalue

def plot_scatter(result_csv_file,input_file_path, solver1, solver2,subadditivity_only=False,log_plot=True):
    """
    Scatter plot which represents the time ratio using two methods, the x_axis is #vertices/#additive vertices.
    """
    allFiles=glob.glob(os.path.join(input_file_path+'*naive.csv'))
    df=pd.read_csv(result_csv_file)
    additive_ratio=[]
    time_ratio=[]
    for f in allFiles:
        with open(f,mode='r') as readfile:
            function = csv.reader(readfile,delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in function:
                name,te,pe,num_bkpts,num_v,num_addv=row[0],row[1],row[2],row[3],row[4],row[5]
                break
        if subadditivity_only and float(QQ(pe))>0:
            continue
        method1,lp1=solver1.split('_')
        method2,lp2=solver2.split('_')
        t1=df.loc[df['name']==name].loc[df['two_epsilon']==te].loc[df['p_epsilon']==pe].loc[df['node_selection']==method1].loc[df['lp_size']==lp1]['time(s)']
        t2=df.loc[df['name']==name].loc[df['two_epsilon']==te].loc[df['p_epsilon']==pe].loc[df['node_selection']==method2].loc[df['lp_size']==lp2]['time(s)']
        if len(t1)>0 and len(t2)>0:
            additive_ratio.append(float(num_v)/float(num_addv))
            time_ratio.append(float(t1)/float(t2))
    plot_df=pd.DataFrame()
    plot_df['additive_ratio']=additive_ratio
    if log_plot:
        plot_df['time_ratio']=[log(t) for t in time_ratio]
    else:
        plot_df['time_ratio']=time_ratio
    return plot_df

def plot_performance_profile_new(result_csv_file, input_file_path,methods,rm,subadditivity_only=False):
    """
    Plot the performance profile based on the results from result_csv_file.
    """
    allFiles=glob.glob(os.path.join(input_file_path+'*naive.csv'))
    df=pd.read_csv(result_csv_file)
    res=[]
    for f in allFiles:
        cur_res=[-1]*len(methods)
        with open(f,mode='r') as readfile:
            function = csv.reader(readfile,delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in function:
                name,te,pe,num_bkpts,num_v,num_addv=row[0],row[1],row[2],row[3],row[4],row[5]
                break
        if subadditivity_only and float(QQ(pe))>0:
            continue
        for i in range(len(methods)):
            col=methods[i]
            if col[:5]=='cplex':
                setting=col[6:]
                t=df.loc[df['name']==name].loc[df['two_epsilon']==te].loc[df['p_epsilon']==pe].loc[df['node_selection']=='cplex'].loc[df['lp_size']==setting]['time(s)']
                if len(t)>0:
                    cur_res[i]=float(t)
            elif col=='naive':
                t=df.loc[df['name']==name].loc[df['two_epsilon']==te].loc[df['p_epsilon']==pe].loc[df['node_selection']=='naive']['time(s)']
                if len(t)>0:
                    cur_res[i]=float(t)
            else:
                method,lp=col.split('_')
                t=df.loc[df['name']==name].loc[df['two_epsilon']==te].loc[df['p_epsilon']==pe].loc[df['node_selection']==method].loc[df['lp_size']==lp]['time(s)']
                if len(t)>0:
                    cur_res[i]=float(t)
        if not cur_res==[-1]*len(methods):
            shortest=min([t for t in cur_res if t>=0])
            cur_ratio=[r/shortest if r>0 else float(rm*10) for r in cur_res]
            res.append(cur_ratio)
    plot_df=pd.DataFrame()
    X=[float(1+i*0.1) for i in range(ZZ((rm-1)*10)+1)]
    plot_df['tau']=X
    for i in range(len(methods)):
        col=methods[i]
        col_ratio=[row[i] for row in res]
        plot_df[col]=[float(counttime(col_ratio,xx)/len(res)) for xx in X]
    return plot_df

def remove_outliers(l):
    """
    Remove outliers from the list l based on Tukey's fences.
    """
    q1=np.percentile(l,25)
    q3=np.percentile(l,75)
    res=[]
    for v in l:
        if q1-3/2*(q3-q1)<=v<=q3+3/2*(q3-q1):
            res.append(v)
    return res

def compute_cv(li,cv=0.2):
    """
    Compute the coefficient of variance (the ratio of standard deviation and the mean).
    """
    # not enough data
    l=remove_outliers(li)
    if len(l)<=2:
        return -1
    mean=sum(l)/len(l)
    var=sum((v-mean)^2 for v in l)/len(l)
    std=sqrt(var)
    return float(std/mean)

def convert_string_to_list_float(string):
    """
    Convert the string of a list to the actual floating point list.
    """
    return [float(l) for l in string[1:-1].split(", ")]

def convert_string_to_list_QQ(string):
    """
    Convert the string of a list to the actual QQ list.
    """
    return [QQ(l) for l in string[1:-1].split(", ")]




