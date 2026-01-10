#!/usr/bin/env python3
"""Генератор дерева структуры проекта для GitHub"""

import os
from pathlib import Path

def should_ignore(path_name, ignore_patterns):
    """Проверяет, нужно ли игнорировать путь"""
    for pattern in ignore_patterns:
        if pattern in path_name:
            return True
    return False

def get_file_icon(path):
    """Возвращает иконку для типа файла"""
    if path.is_dir():
        return "📁"
    ext = path.suffix.lower()
    icons = {
        '.py': '🐍',
        '.md': '📝',
        '.js': '📜',
        '.ts': '📘',
        '.html': '🌐',
        '.css': '🎨',
        '.json': '📋',
        '.yml': '⚙️',
        '.yaml': '⚙️',
        '.toml': '⚙️',
        '.sh': '🔧',
        '.dockerfile': '🐳',
    }
    return icons.get(ext, '📄')

def generate_tree(root_dir, output_file, max_depth=4):
    """Генерирует дерево структуры проекта"""
    
    ignore_patterns = [
        '__pycache__',
        'node_modules',
        '.git',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        '.venv',
        '.next',
        '*.pyc',
        '*.egg-info',
        '*.zip',
        '.DS_Store',
        'dist/',
        'build/',
        '.env',
        '.env.bak',
        'CACHEDIR.TAG',
    ]
    
    lines = []
    lines.append("# 🌳 Структура проекта HEAN\n")
    lines.append("```\n")
    
    root = Path(root_dir)
    
    def tree(dir_path, prefix='', depth=0, is_last=True):
        if depth > max_depth:
            return
        
        name = dir_path.name if dir_path != root else root.name
        
        if should_ignore(name, ignore_patterns):
            return
        
        # Иконка
        icon = get_file_icon(dir_path)
        
        # Соединитель для дерева
        if dir_path == root:
            connector = ''
        else:
            connector = '└── ' if is_last else '├── '
        
        # Формируем строку
        tree_line = f"{prefix}{connector}{icon} {name}"
        if dir_path.is_dir():
            tree_line += "/"
        
        lines.append(tree_line)
        
        # Рекурсия для директорий
        if dir_path.is_dir() and depth < max_depth:
            try:
                items = sorted(
                    [p for p in dir_path.iterdir() 
                     if not should_ignore(p.name, ignore_patterns)],
                    key=lambda x: (x.is_file(), x.name.lower())
                )
                
                for i, item in enumerate(items):
                    is_last_item = (i == len(items) - 1)
                    extension = '    ' if is_last else '│   '
                    new_prefix = prefix + extension
                    tree(item, new_prefix, depth + 1, is_last_item)
            except PermissionError:
                pass
    
    tree(root)
    lines.append("```\n")
    
    # Записываем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Дерево структуры создано: {output_file}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    output_file = script_dir / 'PROJECT_TREE.md'
    generate_tree(script_dir, output_file, max_depth=4)