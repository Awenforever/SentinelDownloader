#!/bin/bash

# --------------------------------------------------------------------
# author: Awenforever
# date: 2025-10-26
# description: Change git log author name and email using git filter-repo
# --------------------------------------------------------------------

# 交互式输入参数
read -rp "请输入原作者姓名: " OLD_NAME
read -rp "请输入修正后的作者姓名: " CORRECT_NAME
read -rp "请输入原作者邮箱: " OLD_EMAIL
read -rp "请输入修正后的邮箱: " CORRECT_EMAIL

# 参数校验
if [[ -z "$OLD_NAME" || -z "$CORRECT_NAME" || -z "$OLD_EMAIL" || -z "$CORRECT_EMAIL" ]]; then
    echo "错误：所有参数都必须提供，不能为空！"
    exit 1
fi

# 执行 git filter-repo
git filter-repo --force --commit-callback "
if commit.author_name == '$OLD_NAME':
    commit.author_name = '$CORRECT_NAME'
    commit.author_email = '$CORRECT_EMAIL'
if commit.committer_name == '$OLD_NAME':
    commit.committer_name = '$CORRECT_NAME'
    commit.committer_email = '$CORRECT_EMAIL'
"