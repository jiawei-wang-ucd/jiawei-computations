import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats; from scipy.stats import sem, t, kurtosis, kurtosistest, skew
from scipy.stats.mstats import gmean

np.random.seed(9001)

def generate_dataframe(path, metric = 'time', stats = 'mean', time_out = 3600):
    """
    Generate a data frame which reports cpu time or memory for one computation task.
    """
    df = pd.DataFrame()
    algorithms = os.listdir(path)
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    for alg in algorithms:
        values = []
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 0:
                if metric == 'time':
                    values.append(time_out)
                elif metric == 'memory':
                    values.append('TimeOut')
                else:
                    raise ValueError
                continue
            if temp_df.shape[0] == 1:
                if metric == 'time':
                    values.append(temp_df['cputime(s)'][0])
                elif metric == 'memory':
                    values.append(temp_df['memory(GB)'][0])
                else:
                    raise ValueError
                continue
            # choose time or memory
            if metric == 'time':
                temps = temp_df['cputime(s)']
            elif metric == 'memory':
                temps = temp_df['memory(GB)']
            else:
                raise ValueError
            
            # choose a type of statistics
            if stats == 'arithmetic_mean':
                values.append(np.mean(temps))
            elif stats == 'mean_bootstrap':
                values.append(np.mean(bootstrap_CI(temps)))
            elif stats == 'CI_bootstrap':
                values.append(bootstrap_CI(temps))
            elif stats == 'mean_t_distribution':
                values.append(np.mean(t_distribution_CI(temps)))
            elif stats == 'CI_t_distribution':
                values.append(t_distribution_CI(temps))
            elif stats == 'max':
                values.append(max(temps))
            else:
                raise ValueError
        df[alg] = values
    return df

def generate_distribution_histogram_one_instance(fname, values, bins = 'rice', title_name = None, title_size = None, xlabel_name = None, xlabel_size = None, ylabel_name = None, ylabel_size = None, **kwargs):
    """
    Generate and save a plot of the histogram of running time distribution.
    The parameter bins represents the method to calculate optimal bin width.
    The plot is stored in fname.
    """
    plt.hist(values, bins = bins, **kwargs)
    if title_name is not None:
        if title_size is not None:
            plt.title(title_name, fontsize = title_size)
        else:
            plt.title(title_name)
    if xlabel_name is not None:
        if xlabel_size is not None:
            plt.xlabel(xlabel_name, fontsize = xlabel_size)
        else:
            plt.xlabel(xlabel_name)
    if ylabel_name is not None:
        if ylabel_size is not None:
            plt.ylabel(ylabel_name, fontsize = ylabel_size)
        else:
            plt.ylabel(ylabel_name)
    plt.savefig(fname)
    plt.close()

def generate_plot_CI_one_group(fname, x_values, y_mean, y_upper, y_lower, title_name = None, title_size = None, xlabel_name = None, xlabel_size = None, ylabel_name = None, ylabel_size = None, x_ticks = None, log_plot = False, **kwargs):
    """
    Generate and save a plot for visualizing confidence intervals of data in one group/algorithm.
    Each value in x_values represents one test instance, and y_mean, y_upper, y_lower represent the mean, lower and upper limit of confidence interval respectively.
    The plot is stored in fname.
    """
    n = len(x_values)
    assert n == len(y_upper)
    assert n == len(y_lower)
    assert n == len(y_mean)
    y_values_lower = copy(y_lower)
    y_values_upper = copy(y_upper)
    y_values_mean = copy(y_mean)
    if log_plot:
        y_values_lower = [float(log(v)) for v in y_values_lower]
        y_values_upper = [float(log(v)) for v in y_values_upper]
        y_values_mean = [float(log(v)) for v in y_values_mean]
    err_lower = [y_values_mean[i] - y_values_lower[i] for i in range(n)]
    err_upper = [y_values_upper[i] - y_values_mean[i] for i in range(n)]

    plt.errorbar(x=x_values, y=y_values_mean, yerr=[err_upper, err_lower], color="black", capsize=3, linestyle="None", marker="s", markersize=3, **kwargs)
    if x_ticks is not None:
        assert n == len(x_ticks)
        plt.xticks(x_values, x_ticks)
    if title_name is not None:
        if title_size is not None:
            plt.title(title_name, fontsize = title_size)
        else:
            plt.title(title_name)
    if xlabel_name is not None:
        if xlabel_size is not None:
            plt.xlabel(xlabel_name, fontsize = xlabel_size)
        else:
            plt.xlabel(xlabel_name)
    if ylabel_name is not None:
        if ylabel_size is not None:
            plt.ylabel(ylabel_name, fontsize = ylabel_size)
        else:
            plt.ylabel(ylabel_name)
    plt.savefig(fname)
    plt.close()

def shifted_geometric_mean(l, s):
    """
    Compute the shifted geometric mean of a list l with shift s.
    """
    if s<0:
        raise ValueError("The shift s should not be negative.")
    return gmean([v+s for v in l])-s

def bootstrap_CI(l, N=1000, alpha=0.95):
    """
    Compute the confidence interval of the sample mean using bootstrap.
    """
    mean_list = [np.mean(np.random.choice(l,len(l),replace=True)) for i in xrange(N)]
    return np.percentile(mean_list,(1-alpha)/2*100), np.percentile(mean_list,(1+alpha)/2*100)

def t_distribution_CI(l, alpha=0.95):
    """
    Compute the confidence interval of the sample mean using t distribution.
    """
    n = len(l)
    m = np.mean(l)
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




