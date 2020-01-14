import git; from git import Repo

branch_algorithm_dictionary = {
'naive': 'naive computation',
'bb': 'branch and bound'
}

branch_parameter_dictionary = {
'master': {'method' : 'branch_bound', 'search_method' : 'DFS', 'lp_size' : 0, 'solver' : 'Coin'},
'naive': {'method' : 'naive', 'search_method' : '', 'lp_size' : None, 'solver' : ''},
'bb': {'method' : 'branch_bound', 'search_method' : 'DFS', 'lp_size' : 0, 'solver' : 'Coin'}
}

def git_branch(path):
    return Repo(path).active_branch.name

