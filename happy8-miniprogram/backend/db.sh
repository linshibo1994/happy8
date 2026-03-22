#!/bin/bash

# 数据库管理脚本
# 用法: ./db.sh [command] [options]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
}

# 安装依赖
install_deps() {
    log_info "安装Python依赖..."
    python3 -m pip install -r requirements.txt
    log_success "依赖安装完成"
}

# 检查数据库连接
check_db() {
    log_info "检查数据库连接..."
    python3 db_manager.py check
    if [ $? -eq 0 ]; then
        log_success "数据库连接正常"
    else
        log_error "数据库连接失败"
        exit 1
    fi
}

# 初始化数据库
init_db() {
    log_info "初始化数据库..."
    python3 db_manager.py init
    if [ $? -eq 0 ]; then
        log_success "数据库初始化完成"
    else
        log_error "数据库初始化失败"
        exit 1
    fi
}

# 创建迁移
create_migration() {
    if [ -z "$2" ]; then
        log_error "请提供迁移消息: ./db.sh migrate '迁移描述'"
        exit 1
    fi
    
    log_info "创建数据库迁移: $2"
    python3 db_manager.py create -m "$2"
    if [ $? -eq 0 ]; then
        log_success "迁移创建完成"
    else
        log_error "迁移创建失败"
        exit 1
    fi
}

# 升级数据库
upgrade_db() {
    local revision=${2:-"head"}
    log_info "升级数据库到版本: $revision"
    python3 db_manager.py upgrade -r "$revision"
    if [ $? -eq 0 ]; then
        log_success "数据库升级完成"
    else
        log_error "数据库升级失败"
        exit 1
    fi
}

# 降级数据库
downgrade_db() {
    local revision=${2:-"-1"}
    log_warning "即将降级数据库到版本: $revision"
    read -p "确定要继续吗? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 db_manager.py downgrade -r "$revision"
        if [ $? -eq 0 ]; then
            log_success "数据库降级完成"
        else
            log_error "数据库降级失败"
            exit 1
        fi
    else
        log_info "降级操作已取消"
    fi
}

# 显示当前版本
show_current() {
    log_info "获取当前数据库版本..."
    python3 db_manager.py current
}

# 显示迁移历史
show_history() {
    log_info "数据库迁移历史:"
    python3 db_manager.py history
}

# 备份数据库
backup_db() {
    local backup_file=${2:-"backup_$(date +%Y%m%d_%H%M%S).sql"}
    log_info "备份数据库到: $backup_file"
    python3 db_manager.py backup -f "$backup_file"
    if [ $? -eq 0 ]; then
        log_success "数据库备份完成: $backup_file"
    else
        log_error "数据库备份失败"
        exit 1
    fi
}

# 恢复数据库
restore_db() {
    if [ -z "$2" ]; then
        log_error "请提供备份文件路径: ./db.sh restore backup.sql"
        exit 1
    fi
    
    if [ ! -f "$2" ]; then
        log_error "备份文件不存在: $2"
        exit 1
    fi
    
    log_warning "即将从备份恢复数据库: $2"
    log_warning "这将覆盖当前所有数据!"
    read -p "确定要继续吗? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 db_manager.py restore -f "$2"
        if [ $? -eq 0 ]; then
            log_success "数据库恢复完成"
        else
            log_error "数据库恢复失败"
            exit 1
        fi
    else
        log_info "恢复操作已取消"
    fi
}

# 验证迁移
validate_migrations() {
    log_info "验证迁移文件..."
    python3 db_manager.py validate
    if [ $? -eq 0 ]; then
        log_success "迁移验证通过"
    else
        log_error "迁移验证失败"
        exit 1
    fi
}

# 完整设置（首次运行）
setup() {
    log_info "开始完整数据库设置..."
    
    # 检查环境
    check_python_env
    
    # 安装依赖
    install_deps
    
    # 初始化数据库
    init_db
    
    # 运行初始化脚本
    log_info "插入初始数据..."
    python3 init_db.py
    if [ $? -eq 0 ]; then
        log_success "初始数据插入完成"
    else
        log_error "初始数据插入失败"
        exit 1
    fi
    
    log_success "数据库设置完成!"
    log_info "现在可以启动API服务:"
    log_info "  python -m uvicorn app.main:app --reload"
}

# 重置数据库（危险操作）
reset_db() {
    log_warning "即将重置数据库!"
    log_warning "这将删除所有数据并重新初始化!"
    read -p "确定要继续吗? 请输入 'RESET' 确认: " -r
    echo
    if [ "$REPLY" = "RESET" ]; then
        log_info "重置数据库..."
        
        # 降级到base
        python3 db_manager.py downgrade -r "base"
        
        # 重新升级
        python3 db_manager.py upgrade
        
        # 插入初始数据
        python3 init_db.py
        
        log_success "数据库重置完成"
    else
        log_info "重置操作已取消"
    fi
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}Happy8小程序数据库管理工具${NC}"
    echo
    echo "用法: ./db.sh [command] [options]"
    echo
    echo "命令:"
    echo "  setup                    完整设置（首次运行）"
    echo "  check                    检查数据库连接"
    echo "  init                     初始化迁移环境"
    echo "  migrate <message>        创建新的迁移"
    echo "  upgrade [revision]       升级数据库（默认: head）"
    echo "  downgrade [revision]     降级数据库（默认: -1）"
    echo "  current                  显示当前版本"
    echo "  history                  显示迁移历史"
    echo "  backup [file]            备份数据库"
    echo "  restore <file>           从备份恢复数据库"
    echo "  validate                 验证迁移文件"
    echo "  reset                    重置数据库（危险操作）"
    echo "  deps                     安装Python依赖"
    echo "  help                     显示此帮助信息"
    echo
    echo "示例:"
    echo "  ./db.sh setup            # 首次完整设置"
    echo "  ./db.sh check            # 检查数据库连接"
    echo "  ./db.sh migrate '添加新字段'"  # 创建迁移"
    echo "  ./db.sh upgrade          # 升级到最新版本"
    echo "  ./db.sh backup           # 备份数据库"
    echo
}

# 主逻辑
case "$1" in
    setup)
        setup
        ;;
    check)
        check_db
        ;;
    init)
        init_db
        ;;
    migrate)
        create_migration "$@"
        ;;
    upgrade)
        upgrade_db "$@"
        ;;
    downgrade)
        downgrade_db "$@"
        ;;
    current)
        show_current
        ;;
    history)
        show_history
        ;;
    backup)
        backup_db "$@"
        ;;
    restore)
        restore_db "$@"
        ;;
    validate)
        validate_migrations
        ;;
    reset)
        reset_db
        ;;
    deps)
        install_deps
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        log_error "未知命令: $1"
        echo
        show_help
        exit 1
        ;;
esac