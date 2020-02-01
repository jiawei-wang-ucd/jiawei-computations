import git; from git import Repo

branch_algorithm_dictionary = {
'naive': 'naive computation',
'bb': 'branch and bound',
'mip': 'MIP formulation'
}

branch_parameter_dictionary = {
'master': {'method' : 'branch_bound', 'search_method' : 'DFS', 'lp_size' : 0, 'solver' : 'cplex'},
'naive': {'method' : 'naive', 'search_method' : '', 'lp_size' : None, 'solver' : ''},
'bb': {'method' : 'branch_bound', 'search_method' : 'DFS', 'lp_size' : 0, 'solver' : 'cplex'},
'mip': {'method' : 'mip', 'search_method' : '', 'lp_size' : None, 'solver' : 'cplex'},
}

def git_branch(path):
    return Repo(path).active_branch.name

