#!/bin/bash

# 数据库初始化脚本

echo "开始初始化数据库..."

# 创建数据库和用户
mysql -u root -p"${MYSQL_ROOT_PASSWORD}" <<EOF
CREATE DATABASE IF NOT EXISTS happy8_miniprogram DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'happy8_user'@'%' IDENTIFIED BY '${MYSQL_PASSWORD}';
GRANT ALL PRIVILEGES ON happy8_miniprogram.* TO 'happy8_user'@'%';
FLUSH PRIVILEGES;
EOF

echo "数据库初始化完成"

# 设置MySQL配置优化
mysql -u root -p"${MYSQL_ROOT_PASSWORD}" <<EOF
SET GLOBAL innodb_buffer_pool_size = 268435456;  -- 256MB
SET GLOBAL max_connections = 200;
SET GLOBAL query_cache_size = 67108864;  -- 64MB
SET GLOBAL tmp_table_size = 67108864;    -- 64MB
SET GLOBAL max_heap_table_size = 67108864;  -- 64MB
EOF

echo "MySQL配置优化完成"