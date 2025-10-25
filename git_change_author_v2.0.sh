#!/bin/bash

read -rp "请输入原作者姓名: " OLD_NAME
read -rp "请输入修正后的作者姓名: " CORRECT_NAME
read -rp "请输入原作者邮箱: " OLD_EMAIL
read -rp "请输入修正后的邮箱: " CORRECT_EMAIL

if [[ -z "$OLD_NAME" || -z "$CORRECT_NAME" || -z "$OLD_EMAIL" || -z "$CORRECT_EMAIL" ]]; then
    echo "错误：所有参数都必须提供，不能为空！"
    exit 1
fi

# 检查是否安装了 git-filter-repo
if ! command -v git-filter-repo &> /dev/null; then
    echo "正在安装 git-filter-repo..."
    pip install git-filter-repo
fi

# 方法1：使用环境变量（推荐）
export OLD_NAME CORRECT_NAME OLD_EMAIL CORRECT_EMAIL

git filter-repo --force --commit-callback "
import os
old_name = os.environ['OLD_NAME']
correct_name = os.environ['CORRECT_NAME']
old_email = os.environ['OLD_EMAIL']
correct_email = os.environ['CORRECT_EMAIL']

if commit.author_name == old_name.encode('utf-8'):
    commit.author_name = correct_name.encode('utf-8')
    commit.author_email = correct_email.encode('utf-8')
if commit.committer_name == old_name.encode('utf-8'):
    commit.committer_name = correct_name.encode('utf-8')
    commit.committer_email = correct_email.encode('utf-8')
"

echo "历史记录修改完成！"