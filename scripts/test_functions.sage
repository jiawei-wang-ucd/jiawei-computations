import gc
import sys
import csv

import resource; from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp

#sys.path.append(os.path.join(os.getcwd(), '..', 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *


def measure_time_resource(function, args=tuple(), kwargs={}):
    '''Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by your
    function/callable by summing `user` and `sys`. `real` is just like the wall
    clock.
    Note that `sys` and `user`'s resolutions are limited by the resolution of
    the operating system's software clock (check `man 7 time` for more
    details).
    '''
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    function(*args, **kwargs)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
    res={'real': end_time - start_time,
        'sys': end_resources.ru_stime - start_resources.ru_stime,
        'user': end_resources.ru_utime - start_resources.ru_utime}
    res['cpu']=res['sys']+res['user']
    res['mem']=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
    return res

def report_performance(function, file_name, iterations, args=tuple(), kwargs={}):
    with open(file_name, mode = 'w') as writefile:
        performance_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_table.writerow(['walltime(s)','cputime(s)','systime(s)','usertime(s)','memory(GB)'])
        writefile.flush()
        for i in range(iterations):
            res = measure_time_resource(function, args=args, kwargs=kwargs)
            performance_table.writerow([res['real'], res['cpu'], res['sys'], res['user'], res['mem']])
            del res
            gc.collect()
    writefile.close()

def minimum_of_delta_pi(fn, method = 'branch_bound', search_method = 'DFS', lp_size = 0, solver = 'Coin'):
    if method == 'naive':
        return minimum_of_delta_pi_naive(fn)
    elif method == 'branch_bound':
        T = SubadditivityTestTree(fn)
        global_min = T.minimum(search_method = search_method, max_number_of_bkpts = lp_size, solver = solver)
        del T
        gc.collect()
        return global_min
    elif method == 'mip':
        if not hasattr(fn, 'mip'):
            fn.mip = generate_mip_of_delta_pi_min_dlog(fn, solver = solver)
        return fn.mip.solve()
    else:
        raise ValueError

def minimum_of_delta_pi_naive(fn):
    """
    Return the min of delta_pi of fn. (Quatratic complexity)
    """
    global_min=10000
    for x in fn.end_points():
        for y in fn.end_points():
            if y<x:
                continue
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            z=1+z
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    return global_min

def generate_mip_of_delta_pi_min_dlog(fn,solver='Coin'):
    """
    Generate the Disaggregated Logarithmic mip formulation of computing the minimum of delta pi.
    """
    bkpts=fn.end_points()
    values=fn.values_at_end_points()
    n=len(bkpts)
    m=ceil(log(n-1,2))
    bkpts2=bkpts+[1+bkpts[i] for i in range(1,n)]
    values2=values+[values[i] for i in range(1,n)]

    p = MixedIntegerLinearProgram(maximization=False, solver=solver)

    xyz = p.new_variable()
    x,y,z = xyz['x'],xyz['y'],xyz['z']
    vxyz = p.new_variable()
    vx,vy,vz = vxyz['vx'],vxyz['vy'],vxyz['vz']
    lambda_x = p.new_variable(nonnegative=True)
    lambda_y = p.new_variable(nonnegative=True)
    lambda_z = p.new_variable(nonnegative=True)
    s_x=p.new_variable(binary=True)
    s_y=p.new_variable(binary=True)
    s_z=p.new_variable(binary=True)
    gamma_x = p.new_variable(nonnegative=True)
    gamma_y = p.new_variable(nonnegative=True)
    gamma_z = p.new_variable(nonnegative=True)

    p.set_objective(vx+vy-vz)

    p.add_constraint(sum([lambda_x[i]*bkpts[i] for i in range(n)])==x)
    p.add_constraint(sum([lambda_y[i]*bkpts[i] for i in range(n)])==y)
    p.add_constraint(sum([lambda_z[i]*bkpts2[i] for i in range(2*n-1)])==z)
    p.add_constraint(x+y==z)
    p.add_constraint(sum([lambda_x[i]*values[i] for i in range(n)])==vx)
    p.add_constraint(sum([lambda_y[i]*values[i] for i in range(n)])==vy)
    p.add_constraint(sum([lambda_z[i]*values2[i] for i in range(2*n-1)])==vz)
    p.add_constraint(sum([lambda_x[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_y[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_z[i] for i in range(2*n-1)])==1)

    for i in range(n):
        p.add_constraint(lambda_x[i]==gamma_x[2*i+1]+gamma_x[2*i])
        p.add_constraint(lambda_y[i]==gamma_y[2*i+1]+gamma_y[2*i])
    for i in range(2*n-1):
        p.add_constraint(lambda_z[i]==gamma_z[2*i+1]+gamma_z[2*i])
        p.add_constraint(gamma_x[0]==gamma_x[2*n-1]==gamma_y[0]==gamma_y[2*n-1]==gamma_z[0]==gamma_z[4*n-3]==0)

    for k in range(m):
        p.add_constraint(sum([(gamma_x[2*i-1]+gamma_x[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_x[k])
        p.add_constraint(sum([(gamma_y[2*i-1]+gamma_y[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_y[k])
    for k in range(m+1):
        p.add_constraint(sum([(gamma_z[2*i-1]+gamma_z[2*i])*int(format(i-1,'0%sb' %(m+1))[k])  for i in range(1,2*n-1)])==s_z[k])

    return p

