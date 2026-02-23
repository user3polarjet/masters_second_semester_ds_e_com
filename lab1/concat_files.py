import pathlib
import os
import typing
import shutil
import asyncio

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

async def main():
    BUILD_DIR.mkdir(exist_ok=True)
    with open(BUILD_DIR / 'task.md', 'w') as outio:
        with open(SCRIPT_DIR / 'lab1_task.md') as fileio:
            outio.write('#lab1_task.md\n')
            outio.writelines(''.join(fileio.readlines()))
        lab_dir = SCRIPT_DIR / 'Lab_work_1'
        for filename in os.listdir(lab_dir):
            filepath = lab_dir / filename
            with open(filepath) as fileio:
                outio.write(f'#{filename}\n')
                outio.writelines(''.join(fileio.readlines()))



if __name__ == '__main__':
    asyncio.run(main())