"""
ms-swift 기반 SFT 학습 스크립트

ms-swift CLI (swift sft) 또는 Python API를 사용하여 Qwen3-8B를 LoRA로 fine-tune합니다.

Usage:
    python -m src.sft.train --config configs/sft/qwen3_8b_fourclass.yaml
    python -m src.sft.train --config configs/sft/qwen3_8b_fourclass.yaml --dry_run
"""

import os
import sys
import yaml
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_swift_binary() -> str:
    """swift CLI 바이너리 경로를 찾습니다."""
    import shutil
    swift_path = shutil.which('swift')
    if swift_path:
        return swift_path
    # 현재 Python 인터프리터의 bin 디렉토리에서 찾기
    python_bin_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(python_bin_dir, 'swift')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("swift binary not found. Install ms-swift: pip install ms-swift")


def build_swift_command(config: dict, data_dir: str = None, output_dir: str = None) -> list:
    """ms-swift CLI 명령어 생성"""
    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, 'data', 'sft')
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'results', 'sft')

    classification_type = config.get('classification_type', 'fourclass')
    train_file = os.path.join(data_dir, f'train_{classification_type}.jsonl')
    val_file = os.path.join(data_dir, f'val_{classification_type}.jsonl')

    swift_bin = find_swift_binary()
    cmd = [
        swift_bin, 'sft',
        '--model', config.get('model', 'Qwen/Qwen3-8B'),
        '--train_type', config.get('train_type', 'lora'),
        '--dataset', train_file,
        '--torch_dtype', config.get('torch_dtype', 'bfloat16'),
        '--num_train_epochs', str(config.get('num_train_epochs', 3)),
        '--per_device_train_batch_size', str(config.get('per_device_train_batch_size', 1)),
        '--gradient_accumulation_steps', str(config.get('gradient_accumulation_steps', 16)),
        '--learning_rate', str(config.get('learning_rate', 1e-4)),
        '--lr_scheduler_type', config.get('lr_scheduler_type', 'cosine'),
        '--max_length', str(config.get('max_length', 8192)),
        '--eval_steps', str(config.get('eval_steps', 50)),
        '--save_steps', str(config.get('save_steps', 50)),
        '--output_dir', output_dir,
        '--logging_steps', str(config.get('logging_steps', 5)),
        '--warmup_ratio', str(config.get('warmup_ratio', 0.05)),
        '--weight_decay', str(config.get('weight_decay', 0.1)),
    ]

    # Gradient checkpointing
    if config.get('gradient_checkpointing', True):
        cmd.extend(['--gradient_checkpointing', 'true'])

    # Save total limit
    if config.get('save_total_limit'):
        cmd.extend(['--save_total_limit', str(config['save_total_limit'])])

    # Loss scale (Qwen3 thinking mode)
    if config.get('loss_scale'):
        cmd.extend(['--loss_scale', config['loss_scale']])

    # LoRA 설정
    if config.get('train_type', 'lora') == 'lora':
        cmd.extend([
            '--lora_rank', str(config.get('lora_rank', 8)),
            '--lora_alpha', str(config.get('lora_alpha', 32)),
            '--target_modules', config.get('target_modules', 'all-linear'),
        ])
        if config.get('lora_dropout') is not None:
            cmd.extend(['--lora_dropout', str(config['lora_dropout'])])

    # DataLoader 최적화
    if config.get('dataloader_num_workers'):
        cmd.extend(['--dataloader_num_workers', str(config['dataloader_num_workers'])])
    if config.get('dataloader_pin_memory', False):
        cmd.extend(['--dataloader_pin_memory', 'true'])

    # Validation 데이터가 있으면 추가
    if os.path.exists(val_file):
        cmd.extend(['--val_dataset', val_file])

    # 모델 다운로드 경로: 환경변수로 설정 (ms-swift v3에서 CLI 인자 미지원)
    # HF_HOME은 run_training()에서 설정

    return cmd


def run_training(config_path: str, data_dir: str = None, output_dir: str = None, dry_run: bool = False):
    """SFT 학습 실행"""
    config = load_config(config_path)

    print(f"{'='*80}")
    print("SFT Training Configuration")
    print(f"{'='*80}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}")

    cmd = build_swift_command(config, data_dir, output_dir)
    cmd_str = ' \\\n  '.join(cmd)

    print(f"\nCommand:")
    print(f"  {cmd_str}")

    if dry_run:
        print("\n[DRY RUN] Command not executed.")
        return

    # 모델 캐시 경로를 환경변수로 설정 (로컬 디스크 대신 NFS 사용)
    env = os.environ.copy()
    if config.get('model_cache_dir'):
        env['HF_HOME'] = config['model_cache_dir']
        env['USE_HF'] = '1'
        env['MODELSCOPE_CACHE'] = os.path.join(config['model_cache_dir'], 'modelscope')

    # 멀티 GPU 분산 학습 (DDP): CUDA_VISIBLE_DEVICES에서 GPU 수 자동 감지
    cuda_devices = env.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices:
        nproc = len(cuda_devices.split(','))
    else:
        import torch
        nproc = torch.cuda.device_count()
    if nproc > 1 and 'NPROC_PER_NODE' not in env:
        env['NPROC_PER_NODE'] = str(nproc)
        print(f"\n[Multi-GPU] NPROC_PER_NODE={nproc} (auto-detected)")

    print(f"\nStarting training...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        print("\nTraining completed successfully!")
    else:
        print(f"\nTraining failed with return code: {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description='ms-swift SFT Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='SFT data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without executing')

    args = parser.parse_args()

    run_training(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
