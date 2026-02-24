import typing
import pickle
import os
import pathlib
import shutil
import argparse
import asyncio
import json

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

def needs_rebuild(output_path: pathlib.Path | str, input_paths: typing.Iterable[pathlib.Path | str]) -> bool:
    if not os.path.exists(output_path):
        res = True
    else:
        output_stat = os.stat(output_path)
        res = any(os.stat(input_path).st_mtime > output_stat.st_mtime for input_path in input_paths)
    if res:
        print(f'build: {output_path}')
    return res

def clean_command():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)

async def seq_cmd(*args: str | pathlib.Path, check: bool=True, **kwargs: typing.Any):
    print('cmd: ', args)
    process = await asyncio.subprocess.create_subprocess_exec(*args, **kwargs)
    await process.communicate()
    if check:
        assert process.returncode == 0

async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')
    parser_clean = subparsers.add_parser('clean')
    parser_clean.set_defaults(func=clean_command)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func()
        return

    lab_work_dir = SCRIPT_DIR / 'Lab_work_8'
    for dirpath_str, dirnames, filenames in os.walk(lab_work_dir):
        dirpath = pathlib.Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            assert len(filepath.suffixes) == 1
            if filepath.suffix in ('.xls', '.xlsx'):
                os.remove(filepath)

if __name__ == '__main__':
    asyncio.run(main())

