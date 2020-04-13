#!/usr/bin/env bash

# checks if branch has something pending
function parse_git_dirty() {
    git diff --quiet --ignore-submodules HEAD 2>/dev/null; [ $? -eq 1 ] && echo "*"
}

# gets the current git branch
function parse_git_branch() {
    git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e "s/* \(.*\)/\1$(parse_git_dirty)/"
}

# get last commit hash prepended with @ (i.e. @8a323d0)
function parse_git_hash() {
    git rev-parse --short HEAD 2> /dev/null | sed "s/\(.*\)/@\1/"
}

# DEMO
echo "Git current branch and commit of the repository and submodules"
echo "Git current branch and commit of the repository and submodules" > "./system_info/git_branch_commit_info"
echo "Git current branch and commit of the repository and submodules" > "./jiawei-computational-results/git_branch_commit_info"

GIT_BRANCH=$(parse_git_branch)$(parse_git_hash)
echo "jiawei-computation:"${GIT_BRANCH}

echo "\njiawei-computation" >> "./system_info/git_branch_commit_info"
echo ${GIT_BRANCH} >> "./system_info/git_branch_commit_info"

echo "\njiawei-computation" >> "./jiawei-computational-results/git_branch_commit_info"
echo ${GIT_BRANCH} >> "./jiawei-computational-results/git_branch_commit_info"

cd jiawei-computational-results

GIT_BRANCH=$(parse_git_branch)$(parse_git_hash)
echo "jiawei-computational-results:"${GIT_BRANCH}

echo "\njiawei-computational-results" >> "../system_info/git_branch_commit_info"
echo ${GIT_BRANCH} >> "../system_info/git_branch_commit_info"

echo "\njiawei-computational-results" >> "git_branch_commit_info"
echo ${GIT_BRANCH} >> "git_branch_commit_info"

cd cutgeneratingfunctionology

GIT_BRANCH=$(parse_git_branch)$(parse_git_hash)
echo "cutgeneratingfunctionology:"${GIT_BRANCH}

echo "\ncutgeneratingfunctionology" >> "../../system_info/git_branch_commit_info"
echo ${GIT_BRANCH} >> "../../system_info/git_branch_commit_info"

echo "\ncutgeneratingfunctionology" >> "../git_branch_commit_info"
echo ${GIT_BRANCH} >> "../git_branch_commit_info"
