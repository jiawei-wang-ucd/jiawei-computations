import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats; from scipy.stats import sem, t, kurtosis, kurtosistest, skew, variation, levene, bartlett
from scipy.stats.mstats import gmean

np.random.seed(9001)

def MLE_fit_rejection_ratio(path, alpha = 0.05, distribution = 'norm'):
    """
    For a given distribution, fit the data and use KS test to verify goodness of fit. Return the ratio of rejections.
    """
    if distribution == 'norm':
        dist = stats.norm
    elif distribution == 'powerlognorm':
        dist = stats.powerlognorm
    elif distribution == 'cauchy':
        dist = stats.cauchy
    elif distribution == 'exponential':
        dist = stats.expon
    else:
        raise ValueError
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    total = 0
    rejections = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            if temp_df.shape[0] == 30:
                total += 1
                cputimes = list(temp_df['cputime(s)'])
                # fit data
                args=dist.fit(cputimes)
                p = stats.kstest(cputimes, dist.cdf, args)[1]
                if p < alpha:
                    rejections += 1
    return rejections, total, rejections * 1.0/total

def different_variance_ratio(path, alpha = 0.05, parametric = False):
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    total = 0
    different_variance = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            if temp_df.shape[0] == 30:
                total += 1
                cputimes = list(temp_df['cputime(s)'])
                walltimes = list(temp_df['walltime(s)'])
                p = p_value_equal_variance(cputimes, walltimes, parametric = parametric, center = 'median')
                if p < alpha:
                    different_variance += 1
    return different_variance, total, different_variance * 1.0/total

def p_value_equal_variance(sample1, sample2, center = 'median', parametric = True):
    """
    Use Leneve's test to test equal variance of two samples with unknown distribution.
    """
    if parametric:
        stat, p = bartlett(sample1, sample2)
    else:
        stat, p = levene(sample1, sample2, center = center)
    return p

def max_range(path, col = 'cputime(s)'):
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    rng = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 30:
                temp_rng = max(temp_df[col])-min(temp_df[col])
                rng = max(temp_rng, rng)
    return rng

def large_wall_clock_time_variance_ratio(path):
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    total = 0
    large_variance = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 30:
                total += 1
                cputime_var = sample_variance(path + '/' + alg + '/' + f + '.csv', col = 'cputime(s)')
                walltime_var = sample_variance(path + '/' + alg + '/' + f + '.csv', col = 'walltime(s)')
                if cputime_var < walltime_var:
                    large_variance += 1
    return large_variance, total, large_variance * 1.0/total

def positive_skew_ratio(path, col = 'cputime(s)'):
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    total = 0
    positive_skew = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 30:
                total += 1
                if sample_skew(path + '/' + alg + '/' + f + '.csv', col = col)>0:
                    positive_skew += 1
    return positive_skew, total, positive_skew * 1.0/total

def positive_kurtosis_ratio(path, col = 'cputime(s)'):
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    total = 0
    positive_kurtosis = 0
    for alg in algorithms:
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 30:
                total += 1
                if sample_kurtosis(path + '/' + alg + '/' + f + '.csv', col = col)>0:
                    positive_kurtosis += 1
    return positive_kurtosis, total, positive_kurtosis * 1.0/total

def sample_variance(result_csv_file, col = 'cputime(s)'):
    df = pd.read_csv(result_csv_file)
    values = df[col]
    return variation(values)

def sample_skew(result_csv_file, col = 'cputime(s)'):
    df = pd.read_csv(result_csv_file)
    values = df[col]
    return skew(values)

def sample_kurtosis(result_csv_file, col = 'cputime(s)'):
    df = pd.read_csv(result_csv_file)
    values = df[col]
    return kurtosis(values)

def qqplot(fname, result_csv_file, col = 'cputime(s)'):
    """
    QQ plot to check normality assumption.
    """
    df = pd.read_csv(result_csv_file)
    values = df[col]
    stats.probplot(values, dist="norm", plot=pylab)
    pylab.title('Q-Q plot')
    pylab.savefig(fname)

def performance_profile_plot(df, time_out = 3600, tau_max = 100, log_scale = True):
    """
    Return a dataframe for plotting performance profile.
    """
    total_instances = df.shape[0]
    best_time = df.min(axis = 1).to_list()
    new_df = pd.DataFrame()
    
    def counttime(l,threshold):
        return sum(1 if v<=threshold else 0 for v in l)
    
    for col in df.columns:
        if log_scale:
            new_df[col] = [math.log(df[col][i]/best_time[i]) if df[col][i] < time_out else math.log(tau_max) for i in range(len(best_time))]
        else:
            new_df[col] = [df[col][i]/best_time[i] if df[col][i] < time_out else tau_max for i in range(len(best_time))]
    
    plot_df = pd.DataFrame()
    if log_scale:
        X=[float(0+i*0.1) for i in range(int(math.log(tau_max)*10+1))]
    else:
        X=[float(1+i*0.1) for i in range((tau_max-1)*10+1)]
    plot_df['tau'] = X
    for col in df.columns:
        plot_df[col] = [float(counttime(new_df[col], xx))/total_instances for xx in X]
    return plot_df

def shifted_gmean_df(df, s):
    """
    Return a data frame which computes shifted geometric mean.
    """
    new_df = pd.DataFrame()
    new_df['metric'] = ["shifted_geometric_mean(shift={})".format(s)]
    for col in df.columns:
        if col == 'instances':
            continue
        new_df[col] = [shifted_geometric_mean(df[col],s = s)]
    return new_df

def generate_dataframe(path, metric = 'time', stats = 'arithmetic_mean', time_out = 3600):
    """
    Generate a data frame which reports cpu time or memory for one computation task.
    """
    df = pd.DataFrame()
    algorithms = [f for f in os.listdir(path) if not f.startswith('.')]
    instance_names = [f[:-4] for f in os.listdir(path+"/"+algorithms[0]) if f.endswith('.csv')]
    df['instances'] = instance_names
    for alg in algorithms:
        values = []
        for f in instance_names:
            temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
            # check if the csv file is empty.
            if temp_df.shape[0] == 0:
                # if empty, check if it is due to out of memory of exceeding time limit.
                log_file_name = path + '/' + alg + '/' + f + '.out'
                v = None
                with open(log_file_name) as mytxt:
                    for line in mytxt:
                        if 'memory' in line:
                            v = 'OutOfMemory'
                            break
                if v is not None:
                    values.append(v)
                    continue
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

def generate_node_count_vs_threshold_plot(fname,fn,use_symmetry=False,**kwargs):
    """
    Generate a plot showing the number of nodes in the tree for different max_number_of_bkpts threshold.
    """
    node_count=[]
    T=SubadditivityTestTree(fn,use_symmetry=use_symmetry)
    m=T.minimum(max_number_of_bkpts=1000000000)
    min_nodes=T.number_of_nodes()
    for i in range(len(fn.end_points()*3)):
        T=SubadditivityTestTree(fn,use_symmetry=use_symmetry)
        m=T.minimum(max_number_of_bkpts=i)
        nodes=T.number_of_nodes()
        node_count.append(nodes)
        if nodes == min_nodes:
            break
    plt.plot(node_count, **kwargs)
    plt.xlabel('threshold for using LP estimators')
    plt.ylabel('number of nodes in the tree')
    plt.savefig(fname)
    plt.close()

def generate_benchmark_set_distribution(fname, file_path, bins = 'auto', **kwargs):
    """
    Generate and save a plot of the histogram of number of bkpts (sqrt scale) in the benchmark set.
    """
    all_files=glob.glob(file_path+'/*sobj')
    bkpts=[round(sqrt(len(load(fi).end_points())),2) for fi in all_files]
    plt.hist(bkpts, bins = bins, **kwargs)
    plt.xlabel('square root of total breakpoints')
    plt.ylabel('number of functions')
    plt.savefig(fname)
    plt.close()

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

def generate_plot_CI_one_group(fname, x_values, y_mean, y_upper, y_lower, color = "black", title_name = None, title_size = None, xlabel_name = None, xlabel_size = None, ylabel_name = None, ylabel_size = None, x_ticks = None, log_plot = False, **kwargs):
    """
    Generate and save a plot for visualizing confidence intervals of data in one group/algorithm.
    Each value in x_values represents one test instance, and y_mean, y_upper, y_lower represent the mean, lower and upper limit of confidence interval respectively.
    The plot is stored in fname.
    """
    n = len(x_values)
    assert n == len(y_upper)
    assert n == len(y_lower)
    assert n == len(y_mean)
    y_values_lower = y_lower[:]
    y_values_upper = y_upper[:]
    y_values_mean = y_mean[:]
    if log_plot:
        y_values_lower = [float(log(v)) for v in y_values_lower]
        y_values_upper = [float(log(v)) for v in y_values_upper]
        y_values_mean = [float(log(v)) for v in y_values_mean]
    err_lower = [y_values_mean[i] - y_values_lower[i] for i in range(n)]
    err_upper = [y_values_upper[i] - y_values_mean[i] for i in range(n)]

    plt.errorbar(x=x_values, y=y_values_mean, yerr=[err_upper, err_lower], color=color, capsize=3, linestyle="None", marker="s", markersize=3, **kwargs)
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

def shifted_geometric_mean(l, s, limit = 3600):
    """
    Compute the shifted geometric mean of a list l with shift s.
    """
    if s<0:
        raise ValueError("The shift s should not be negative.")
    temp_l = [v if not isinstance(v, str) else limit for v in l]
    return gmean([v+s for v in temp_l])-s

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

def kurtosis_test_p_value(l):
    """
    Hypothesis test to compute the p value under the null hypothesis that the samples are drawn from a normal distribution.
    """
    if len(l)<20:
        raise ValueError("Not enough data")
    res = kurtosistest(l)
    return res.pvalue

def plot_scatter(df, metric = 'time', time_out = 3600, memory_limit = 8):
    """
    Scatter plot which represents the number of breakpoints vs time/memory.
    """
    instances_info = pd.read_csv('./test_instances/test_instances_info.csv')
    combined_df = df.merge(instances_info, left_on='instances', right_on='file name')
    if metric == 'time':
        for col in combined_df.columns:
            combined_df[col] = [time_out if (v == 'TimeOut' or v == 'OutOfMemory') else v for v in combined_df[col]]
    elif metric == 'memory':
        for col in combined_df.columns:
            combined_df[col] = [memory_limit if (v == 'TimeOut' or v == 'OutOfMemory') else v for v in combined_df[col]]
    else:
        raise ValueError
    #plt.savefig(fname)
    return combined_df

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




