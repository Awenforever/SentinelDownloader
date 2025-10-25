#!/bin/bash

read -rp "请输入原作者姓名: " OLD_NAME
read -rp "请输入修正后的作者姓名: " CORRECT_NAME
read -rp "请输入原作者邮箱: " OLD_EMAIL
read -rp "请输入修正后的邮箱: " CORRECT_EMAIL

if [[ -z "$OLD_NAME" || -z "$CORRECT_NAME" || -z "$OLD_EMAIL" || -z "$CORRECT_EMAIL" ]]; then
    echo "错误：所有参数都必须提供，不能为空！"
    exit 1
fi

git filter-repo --force --commit-callback '
if commit.author_name == "'"$OLD_NAME"'":
    commit.author_name = "'"$CORRECT_NAME"'"
    commit.author_email = "'"$CORRECT_EMAIL"'"
if commit.committer_name == "'"$OLD_NAME"'":
    commit.committer_name = "'"$CORRECT_NAME"'"
    commit.committer_email = "'"$CORRECT_EMAIL"'"
'