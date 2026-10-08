"""导出/复核P0小工件；普通Python可用，Windows复核不需要Isaac Lab。"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    file = root/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/new_stomach_rl_review_bundle.py'
    spec = importlib.util.spec_from_file_location('review_bundle', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['export','verify','audit'])
    parser.add_argument('--source_inventory', type=Path)
    parser.add_argument('--output_directory', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'export':
        if args.source_inventory is None:
            parser.error('--source_inventory必填，不猜测旧工作树路径')
        result = module.export_bundle(args.source_inventory, args.output_directory, module.INVENTORY_SHA)
    elif args.operation=='verify':
        result = module.verify_bundle(args.output_directory, module.INVENTORY_SHA)
    else:
        module.verify_bundle(args.output_directory, module.INVENTORY_SHA)
        print(json.dumps(module.audit_capacity_bundle(args.output_directory),indent=2))
        return
    print(json.dumps(dict(status=result['status'], audited_head=result['audited_head'],
        artifacts=len(result['artifacts']), output=str(args.output_directory.resolve())), indent=2))


if __name__ == '__main__':
    main()
