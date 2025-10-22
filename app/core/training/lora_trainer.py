import subprocess
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.mps

from app.core.config import get_config

def get_base_model_configs():
    config = get_config()
    return config.model_configs


class LoRATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_process = None

        self.training_status = {
            "is_training": False,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get("epochs", 10),
            "loss": None,
            "loss_history": [],
            "error": None,
            "start_time": None,
            "estimated_completion": None,
            "model_name": config.get("modelName", "untitled"),
            "device_info": None,
            "current_step": 0,
            "global_step": 0,
            "checkpoints_saved": 0,
        }

    def _validate_dataset_before_training(self) -> Dict[str, Any]:
        try:
            from app.core.config import get_config
            config = get_config()
            data_dir = config.get_path("data")
            
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_files.extend(list(data_dir.glob(f"*{ext}")))
            
            file_count = len(audio_files)
            batch_size = self.config.get("batchSize", 4)
            
            if file_count == 0:
                return {
                    "valid": False,
                    "error": "No audio files found in the data directory. Please upload some audio files first."
                }
            
            if file_count < batch_size:
                return {
                    "valid": False,
                    "error": f"Insufficient audio files for training. Found {file_count} files but batch size is {batch_size}. Either add more audio files (recommended) or reduce batch size to {file_count}."
                }
            
            metadata_json = data_dir / "metadata.json"
            
            if not metadata_json.exists():
                return {
                    "valid": False,
                    "error": "No metadata found. Please ensure your audio files have associated prompts/descriptions."
                }
            
            steps_per_epoch = max(1, file_count // batch_size)
            total_epochs = self.config.get("epochs", 10)
            total_steps = steps_per_epoch * total_epochs
            checkpoint_every = self.config.get("checkpointSteps", 100)
            
            checkpoint_warning = None
            if total_steps < checkpoint_every:
                checkpoint_warning = (
                    f"WARNING: NOT ENOUGH DATA! "
                    f"Training will complete in {total_steps} steps, but checkpoints are set to save every {checkpoint_every} steps. "
                    f"Add more audio files or reduce checkpoint interval."
                )
            
            return {
                "valid": True,
                "file_count": file_count,
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "checkpoint_every": checkpoint_every,
                "checkpoint_warning": checkpoint_warning
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Dataset validation error: {str(e)}"
            }

    def start_training(self) -> Dict[str, Any]:
        if self.training_status["is_training"]:
            return {"error": "Training already in progress"}

        print("\n" + "="*80)
        print("SMART DEVICE DETECTION")
        print("="*80)

        device_info = self._detect_best_device()
        print(f"SELECTED DEVICE: {device_info['device']}")
        print(f"- Type: {device_info['type']}")
        print(f"- Memory: {device_info['memory_gb']:.2f} GB")
        print(f"- Reason: {device_info['reason']}")

        if torch.backends.mps.is_available() and device_info['type'] == 'mps':
            torch.mps.empty_cache()
        elif torch.cuda.is_available() and device_info['type'] == 'cuda':
            torch.cuda.empty_cache()

        print("\n" + "="*80)
        print("STARTING TRAINING - MODEL SELECTION")
        print("="*80)

        base_model = self.config.get("baseModel", "stable-audio-open-small")
        print(f"RECEIVED TRAINING CONFIG:")
        print(f"- Model Name: {self.config.get('modelName', 'untitled')}")
        print(f"- Base Model: {base_model}")
        print(f"- Epochs: {self.config.get('epochs', 3)}")
        print(f"- Batch Size: {self.config.get('batchSize', 1)}")
        print(f"- Learning Rate: {self.config.get('learningRate', 1e-4)}")
        base_model_configs = get_base_model_configs()

        if base_model not in base_model_configs:
            print(f"ERROR: Unknown base model '{base_model}'")
            print(f"Available models: {list(base_model_configs.keys())}")
            print(f"Defaulting to 'stable-audio-open-small' for safety")
            base_model = "stable-audio-open-small"

        model_info = base_model_configs[base_model]
        print(f"\nMODEL CONFIGURATION:")
        print(f"- Selected Model: {base_model}")
        print(f"- Config File: {model_info['config']}")
        print(f"- Checkpoint File: {model_info['ckpt']}")

        if base_model == "stable-audio-open-1.0":
            print(f"\nWARNING: Using LARGE model (stable-audio-open-1.0)")
            print(f"- This requires significant GPU memory (~8-12 GB)")
            print(f"- Consider using 'stable-audio-open-small' for lower memory usage")
            print(f"- Model size: ~4.52 GB checkpoint")
        else:
            print(f"\nUsing SMALL model (stable-audio-open-small)")
            print(f"- Optimized for lower memory usage (~4-6 GB)")
            print(f"- Model size: ~1.56 GB checkpoint")
            print(f"- Recommended for Apple Silicon Macs")

        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_name = self.config["modelName"]
            model_config = str(project_root / model_info["config"])
            pretrained_ckpt = str(project_root / model_info["ckpt"])

            print(f"\nFILE VALIDATION:")

            config_path = Path(model_config)
            if not config_path.exists():
                error_msg = f"CRITICAL ERROR: Model config file not found: {model_config}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                print(f"Config file exists: {config_path}")
                print(f"Config file size: {config_path.stat().st_size:,} bytes")

            ckpt_path = Path(pretrained_ckpt)
            if not ckpt_path.exists():
                error_msg = f"CRITICAL ERROR: Model checkpoint file not found: {pretrained_ckpt}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                print(f"Checkpoint file exists: {ckpt_path}")
                print(f"Checkpoint file size: {ckpt_path.stat().st_size / (1024**3):.2f} GB")

            print(f"\nFINAL MODEL SELECTION:")
            print(f"- Base Model: {base_model}")
            print(f"- Model Config Path: {model_config}")
            print(f"- Pretrained Checkpoint: {pretrained_ckpt}")

            if self.config.get("modelConfigPath"):
                model_config = str(
                    project_root / self.config["modelConfigPath"])
                print(f"OVERRIDE: Config path overridden to: {model_config}")
            if self.config.get("pretrainedCkptPath"):
                pretrained_ckpt = str(
                    project_root / self.config["pretrainedCkptPath"])
                print(f"OVERRIDE: Checkpoint path overridden to: {pretrained_ckpt}")

            learning_rate = self.config.get("learningRate", 1e-4)
            print(f"LEARNING RATE: {learning_rate}")

            import json
            config_path = Path(model_config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                if 'training' in config_data and 'optimizer_configs' in config_data['training']:
                    if 'diffusion' in config_data['training']['optimizer_configs']:
                        if 'optimizer' in config_data['training']['optimizer_configs']['diffusion']:
                            if 'config' in config_data['training']['optimizer_configs']['diffusion']['optimizer']:
                                old_lr = config_data['training']['optimizer_configs']['diffusion']['optimizer']['config']['lr']
                                config_data['training']['optimizer_configs']['diffusion']['optimizer']['config']['lr'] = learning_rate
                                print(f"Updated learning rate from {old_lr} to {learning_rate} in model config")

                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
                print(f"Updated model config saved to: {config_path}")
            else:
                print(f"WARNING: Model config file not found: {config_path}")

            config = get_config()
            dataset_config = config.get_dataset_config_path()
            save_dir = str(config.get_path("models_fine_tuned") / model_name)
            os.makedirs(save_dir, exist_ok=True)

            if device_info['type'] == 'cpu':
                requested_batch_size = self.config.get("batchSize", 1)
                batch_size = min(requested_batch_size, 1)
                accum_batches = max(1, requested_batch_size // batch_size)
                precision = "32"
                num_workers = 0
                print(f"\nMEMORY SETTINGS (CPU Training):")
                print(f"- Batch Size: {batch_size}")
                print(f"- Gradient Accumulation: {accum_batches}")
                print(f"- Effective Batch Size: {batch_size * accum_batches}")
                print(f"- Precision: {precision}")
                print(f"- Workers: {num_workers}")

            elif base_model == "stable-audio-open-1.0":
                requested_batch_size = self.config.get("batchSize", 1)
                batch_size = min(requested_batch_size, 1)
                accum_batches = max(1, requested_batch_size // batch_size)
                precision = "32"
                num_workers = 0
                print(f"\nMEMORY SETTINGS (Large Model on {device_info['type'].upper()}):")
                print(f"- Batch Size: {batch_size}")
                print(f"- Gradient Accumulation: {accum_batches}")
                print(f"- Effective Batch Size: {batch_size * accum_batches}")
                print(f"- Precision: {precision}")
                print(f"- Workers: {num_workers}")
            else:
                requested_batch_size = self.config.get("batchSize", 1)
                batch_size = min(requested_batch_size, 2)
                accum_batches = max(1, requested_batch_size // batch_size)
                precision = "32"
                num_workers = 0
                print(f"\nMEMORY SETTINGS (Small Model on {device_info['type'].upper()}):")
                print(f"- Batch Size: {batch_size}")
                print(f"- Gradient Accumulation: {accum_batches}")
                print(f"- Effective Batch Size: {batch_size * accum_batches}")
                print(f"- Precision: {precision}")
                print(f"- Workers: {num_workers}")

            memory_flags = [
                "--precision", precision,
                "--accum-batches", str(accum_batches),
                "--gradient-clip-val", "1.0",
                "--num-workers", str(num_workers),
                "--logger", "none",
                "--seed", "42",
            ]

            if device_info['type'] == 'cpu':
                memory_flags.extend([
                    "--strategy", "auto",
                ])
            else:
                memory_flags.extend([
                    "--strategy", "auto",
                ])

            config = get_config()
            venv_python = str(config.get_path("venv") / "bin" / "python")
            if not os.path.exists(venv_python):
                venv_python = sys.executable

            cmd = [
                venv_python, "train.py",
                "--pretrained-ckpt-path", pretrained_ckpt,
                "--model-config", model_config,
                "--dataset-config", dataset_config,
                "--name", model_name,
                "--save-dir", save_dir,
                "--checkpoint-every", str(self.config.get("checkpointSteps", 25)),
                "--batch-size", str(batch_size),
            ] + memory_flags

            print(f"\nTRAINING COMMAND:")
            print(f"{' '.join(cmd)}")
            print(f"\nSAVE DIRECTORY: {save_dir}")

            env = os.environ.copy()
            env['WANDB_MODE'] = 'disabled'
            env['WANDB_SILENT'] = 'true'
            env['WANDB_DISABLED'] = 'true'

            if device_info['type'] == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
                env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                print(f"\nCPU TRAINING SETTINGS:")
                print(f"- CUDA disabled")
                print(f"- MPS disabled")
            elif device_info['type'] == 'mps':
                env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(
                    device_info['memory_ratio'])
                env['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = str(
                    max(0.1, device_info['memory_ratio'] - 0.2))
                print(f"\nMPS MEMORY SETTINGS:")
                print(f"- High Watermark: {device_info['memory_ratio']*100:.0f}%")
                print(f"- Low Watermark: {max(0.1, device_info['memory_ratio'] - 0.2)*100:.0f}%")
            elif device_info['type'] == 'cuda':
                print(f"\nCUDA MEMORY SETTINGS:")
                print(f"- Available: {device_info['memory_gb']:.2f} GB")
                print(f"- Using: {device_info['memory_ratio']*100:.0f}%")

            validation_result = self._validate_dataset_before_training()
            if not validation_result["valid"]:
                print(f"TRAINING VALIDATION FAILED!")
                print(f"{validation_result['error']}")
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "validation_failed": True
                }
            
            if validation_result.get("checkpoint_warning"):
                print(f"\nCHECKPOINT WARNING:")
                print(f"{validation_result['checkpoint_warning']}")
                print(f"Training Stats:")
                print(f"- Audio files: {validation_result['file_count']}")
                print(f"- Batch size: {validation_result['batch_size']}")
                print(f"- Steps per epoch: {validation_result['steps_per_epoch']}")
                print(f"- Total epochs: {validation_result['total_steps'] // validation_result['steps_per_epoch']}")
                print(f"- Total steps: {validation_result['total_steps']}")
                print(f"- Checkpoint every: {validation_result['checkpoint_every']} steps")
                print(f"Recommended checkpoint interval: {max(10, validation_result['total_steps'] // 2)} steps")
                
                return {
                    "success": False,
                    "error": validation_result["checkpoint_warning"],
                    "checkpoint_warning": True,
                    "validation_stats": validation_result
                }

            print(f"\nSTARTING TRAINING PROCESS...")
            print("="*80)

            config = get_config()
            self.training_process = subprocess.Popen(
                cmd,
                cwd=str(config.get_path("stable_audio_tools")),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            self.training_start_time = time.time()  # Store for loss history timing

            current_progress = self.training_status.get("progress", 0)
            current_epoch = self.training_status.get("current_epoch", 0)
            current_step = self.training_status.get("current_step", 0)
            global_step = self.training_status.get("global_step", 0)

            self.training_status.update({
                "is_training": True,
                "start_time": self.training_start_time,
                "progress": current_progress,
                "current_epoch": current_epoch,
                "current_step": current_step,
                "global_step": global_step,
                "checkpoints_saved": 0,
                "error": None,
                "device_info": device_info,
                "total_steps_per_epoch": None,
                "error_messages": []
            })

            monitor_thread = threading.Thread(target=self._monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()

            return {
                "status": "started",
                "message": f"Training started successfully with {base_model} model on {device_info['type'].upper()}",
                "config": self.config,
                "model_info": {
                    "base_model": base_model,
                    "config_file": model_config,
                    "checkpoint_file": pretrained_ckpt
                },
                "device_info": device_info
            }
        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            print(f"\n ERROR: {error_msg}")
            print("="*80)
            self.training_status["error"] = error_msg
            return {"error": error_msg}

    def _detect_best_device(self) -> Dict[str, Any]:
        try:
            if torch.cuda.is_available():
                try:
                    cuda_memory = torch.cuda.get_device_properties(
                        0).total_memory / (1024**3)
                    cuda_capability = torch.cuda.get_device_capability(0)
                    print(f"CUDA available: {cuda_memory:.2f} GB, capability: {cuda_capability}")

                    if cuda_capability[0] > 12 or (cuda_capability[0] == 12 and cuda_capability[1] > 8):
                        print(f"CUDA device too new (capability {cuda_capability}), using CPU")
                        return {
                            "device": "cpu",
                            "type": "cpu",
                            "memory_gb": 0,
                            "memory_ratio": 0,
                            "reason": f"CUDA device too new (capability {cuda_capability}), using CPU"
                        }

                    if cuda_memory >= 16:
                        return {
                            "device": "cuda:0",
                            "type": "cuda",
                            "memory_gb": cuda_memory,
                            "memory_ratio": 0.8,
                            "reason": f"CUDA with {cuda_memory:.2f} GB available"
                        }
                    elif cuda_memory >= 8:
                        return {
                            "device": "cuda:0",
                            "type": "cuda",
                            "memory_gb": cuda_memory,
                            "memory_ratio": 0.6,
                            "reason": f"CUDA with {cuda_memory:.2f} GB available (limited)"
                        }
                except Exception as e:
                    print(f"CUDA check failed: {e}, using CPU")
                    return {
                        "device": "cpu",
                        "type": "cpu",
                        "memory_gb": 0,
                        "memory_ratio": 0,
                        "reason": f"CUDA check failed: {e}, using CPU"
                    }

            if torch.backends.mps.is_available():
                try:
                    mps_allocated = torch.mps.current_allocated_memory() / (1024**3)
                    mps_reserved = torch.mps.driver_allocated_memory() / (1024**3)
                    mps_total = mps_reserved

                    print(f"MPS available: {mps_total:.2f} GB (allocated: {mps_allocated:.2f} GB)")

                    if mps_total >= 8:
                        return {
                            "device": "mps",
                            "type": "mps",
                            "memory_gb": mps_total,
                            "memory_ratio": 0.7,
                            "reason": f"MPS with {mps_total:.2f} GB available"
                        }
                    elif mps_total >= 4:
                        return {
                            "device": "mps",
                            "type": "mps",
                            "memory_gb": mps_total,
                            "memory_ratio": 0.5,
                            "reason": f"MPS with {mps_total:.2f} GB available (limited)"
                        }
                except Exception as e:
                    print(f"MPS memory check failed: {e}")
                    return {
                        "device": "cpu",
                        "type": "cpu",
                        "memory_gb": 0,
                        "memory_ratio": 0,
                        "reason": "MPS available but memory check failed, using CPU for safety"
                    }

            print("No suitable GPU found, using CPU")
            return {
                "device": "cpu",
                "type": "cpu",
                "memory_gb": 0,
                "memory_ratio": 0,
                "reason": "No suitable GPU with sufficient memory, using CPU"
            }

        except Exception as e:
            print(f"Device detection error: {e}, falling back to CPU")
            return {
                "device": "cpu",
                "type": "cpu",
                "memory_gb": 0,
                "memory_ratio": 0,
                "reason": f"Device detection failed: {e}, using CPU"
            }

    def _monitor_training(self):
        try:
            config = get_config()
            save_dir = str(config.get_path(
                "models_fine_tuned") / self.config['modelName'])
            start_time = time.time()
            last_step = 0
            last_loss = None
            last_checkpoint_count = 0

            print(f"\nREAL-TIME TRAINING MONITOR")
            print(f"Save directory: {save_dir}")
            print(f"Monitoring started at: {time.strftime('%H:%M:%S')}")
            print("="*60)

            while self.training_process and self.training_process.poll() is None:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if self.training_process.stdout:
                    try:
                        import select
                        if select.select([self.training_process.stdout], [], [], 0.1)[0]:
                            line = self.training_process.stdout.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    self._process_training_output(
                                        line, elapsed_time)
                    except Exception as e:
                        print(f"Stdout read error: {e}")

                try:
                    if self.training_process.stdout:
                        import select
                        if select.select([self.training_process.stdout], [], [], 0)[0]:
                            line = self.training_process.stdout.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    self._process_training_output(
                                        line, elapsed_time)
                except:
                    pass

                if self.training_process.stderr:
                    try:
                        import select
                        if select.select([self.training_process.stderr], [], [], 0.1)[0]:
                            line = self.training_process.stderr.readline()
                            if line:
                                line = line.strip()
                                print(f"Stderr: {line}")
                                if "error" in line.lower() or "exception" in line.lower() or "warning" in line.lower():
                                    self.training_status["error_messages"].append(
                                        line)
                    except:
                        pass

                if os.path.exists(save_dir):
                    checkpoint_files = list(Path(save_dir).glob("*.ckpt"))
                    current_checkpoint_count = len(checkpoint_files)

                    if current_checkpoint_count > last_checkpoint_count:
                        print(f"CHECKPOINT SAVED! Total: {current_checkpoint_count}")

                        last_checkpoint_count = current_checkpoint_count
                        self.training_status["checkpoints_saved"] = current_checkpoint_count

                        total_epochs = self.config.get("epochs", 10)
                        checkpoint_progress = min(
                            90, (current_checkpoint_count / max(1, total_epochs)) * 90)
                        self.training_status["progress"] = int(
                            checkpoint_progress)
                        print(f"Progress: {int(checkpoint_progress)}% ({current_checkpoint_count} checkpoints)")

                if last_checkpoint_count == 0 and self.training_status["current_step"] == 0:
                    estimated_epoch_time = 600
                    total_epochs = self.config.get("epochs", 10)
                    estimated_total_time = estimated_epoch_time * total_epochs
                    time_progress = min(
                        50, (elapsed_time / estimated_total_time) * 50)
                    self.training_status["progress"] = int(time_progress)
                    print(f"Time-based progress: {int(time_progress)}% (elapsed: {elapsed_time:.1f}s)")

                self.training_status["checkpoints_saved"] = last_checkpoint_count

                if int(elapsed_time) % 10 == 0 and int(elapsed_time) > 0 and self.training_status["current_step"] == 0:
                    debug_step = int(elapsed_time / 10)
                    self.training_status["current_step"] = debug_step
                    self.training_status["current_epoch"] = 0

                if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:
                    self._print_status_update(
                        elapsed_time, last_step, last_loss, last_checkpoint_count)

                time.sleep(1)

            if self.training_process:
                return_code = self.training_process.returncode
                
                training_actually_succeeded = (
                    return_code == 0 and 
                    elapsed_time > 30 and
                    (last_checkpoint_count > 0 or last_loss is not None)
                )
                
                if training_actually_succeeded:
                    print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
                    print(f"Final Stats:")
                    print(f"- Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                    print(f"- Checkpoints saved: {last_checkpoint_count}")
                    print(f"- Final loss: {last_loss if last_loss else 'N/A'}")

                    self.training_status.update({
                        "is_training": False,
                        "progress": 100,
                        "current_epoch": self.config.get("epochs", 10),
                        "loss": last_loss
                    })

                    print(f"\nCheckpoints can be manually unwrapped from the Generation page")
                else:
                    if elapsed_time < 30 and return_code == 0:
                        error_msg = "Training failed due to insufficient training data. You need at least as many audio files as your batch size. Current batch size is {}. Please add more audio files or reduce the batch size.".format(
                            self.config.get("batchSize", 4)
                        )
                        failure_reason = "INSUFFICIENT_DATA"
                    else:
                        try:
                            stdout, stderr = self.training_process.communicate(
                                timeout=5)
                            error_msg = stderr if stderr else stdout
                            if not error_msg:
                                error_msg = "Training process exited unexpectedly"
                        except:
                            error_msg = "Could not capture error output"
                        failure_reason = "UNKNOWN_ERROR"

                    print(f"\nTRAINING FAILED!")
                    print(f"Reason: {failure_reason}")
                    print(f"Duration: {elapsed_time:.1f} seconds")
                    print(f"Return code: {return_code}")
                    print(f"Details: {error_msg}")

                    self.training_status.update({
                        "is_training": False,
                        "error": error_msg,
                        "failure_reason": failure_reason
                    })
        except Exception as e:
            print(f"Monitoring error: {str(e)}")
            self.training_status.update({
                "is_training": False,
                "error": f"Monitoring error: {str(e)}"
            })

    def _process_training_output(self, line: str, elapsed_time: float):
        line_lower = line.lower()

        if "step" in line_lower or "epoch" in line_lower or "it" in line_lower:
            import re

            lightning_pattern = r'epoch\s+(\d+).*?(\d+)/(\d+).*?loss=([0-9.]+)'
            lightning_match = re.search(lightning_pattern, line_lower)

            if lightning_match:
                epoch = int(lightning_match.group(1))
                current_step = int(lightning_match.group(2))
                total_steps = int(lightning_match.group(3))
                loss = float(lightning_match.group(4))

                if (current_step != self.training_status.get("current_step", -1) or 
                    epoch != self.training_status.get("current_epoch", -1)):
                    print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}, Step {current_step}/{total_steps}, Loss: {loss:.4f}")

                self.training_status["current_step"] = current_step
                self.training_status["current_epoch"] = epoch
                self.training_status["loss"] = loss

                total_epochs = self.config.get("epochs", 10)
                global_step = (epoch * total_steps) + current_step
                total_global_steps = total_epochs * total_steps

                step_progress = min(
                    90, (global_step / total_global_steps) * 90)

                self.training_status["global_step"] = global_step
                self.training_status["progress"] = int(step_progress)

                if current_step % 10 == 0 or current_step == 1 or current_step == total_steps:
                    print(f"Global Step: {global_step}, Progress: {step_progress:.1f}%")

                self.training_status["total_steps_per_epoch"] = total_steps

                if epoch >= total_epochs:
                    print(f"\nTARGET EPOCHS REACHED: {epoch}/{total_epochs}")
                    print(f"Automatically stopping training...")
                    if self.training_process and self.training_process.poll() is None:
                        self.training_process.terminate()
                        print(f"Training process terminated successfully")
                        self.training_status.update({
                            "is_training": False,
                            "progress": 100,
                            "current_epoch": epoch,
                            "loss": loss
                        })
                        return

                current_time = time.time()
                if hasattr(self, 'training_start_time'):
                    elapsed = current_time - self.training_start_time
                else:
                    elapsed = 0

                loss_entry = {
                    "time": elapsed,
                    "loss": loss,
                    "step": current_step,
                    "epoch": epoch
                }
                self.training_status["loss_history"].append(loss_entry)

                # Keep only last 1000 points to prevent memory issues
                if len(self.training_status["loss_history"]) > 1000:
                    self.training_status["loss_history"] = self.training_status["loss_history"][-1000:]

                return

            if not self.training_status.get("total_steps_per_epoch"):
                step_patterns = [
                    r'step\s+(\d+)',
                    r'global\s+step\s+(\d+)',
                    r'step\s+(\d+)\s*/\s*(\d+)',
                    r'(\d+)\s*/\s*(\d+)\s*steps',
                    r'step\s*(\d+)',
                    r'(\d+)\s*step',
                ]

                step = None
                epoch = self.training_status.get("current_epoch", 0)

                for pattern in step_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        if len(match.groups()) == 1:
                            step = int(match.group(1))
                        elif len(match.groups()) == 2:
                            step = int(match.group(1))
                        break

                epoch_patterns = [
                    r'epoch\s+(\d+)',
                    r'epoch\s*(\d+)',
                    r'(\d+)\s*epoch',
                ]

                for pattern in epoch_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        epoch = int(match.group(1))
                        break

                if step is not None:
                    print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}, Step {step}")

                    self.training_status["current_step"] = step
                    self.training_status["current_epoch"] = epoch

                    total_epochs = self.config.get("epochs", 10)
                    stored_total_steps = self.training_status.get(
                        "total_steps_per_epoch")

                    if stored_total_steps:
                        global_step = (epoch * stored_total_steps) + step
                        total_global_steps = total_epochs * stored_total_steps
                        step_progress = min(
                            90, (global_step / total_global_steps) * 90)
                    else:
                        estimated_steps_per_epoch = 35
                        global_step = (
                            epoch * estimated_steps_per_epoch) + step
                        total_estimated_steps = total_epochs * estimated_steps_per_epoch
                        step_progress = min(
                            90, (global_step / total_estimated_steps) * 90)

                    self.training_status["global_step"] = global_step
                    self.training_status["progress"] = int(step_progress)

        elif "loss" in line_lower and "=" in line_lower:
            import re
            loss_match = re.search(r'loss[=:\s]*([0-9.]+)', line_lower)
            if loss_match:
                loss = float(loss_match.group(1))
                print(f"[{time.strftime('%H:%M:%S')}] Loss: {loss:.4f}")
                self.training_status["loss"] = loss

                current_time = time.time()
                if hasattr(self, 'training_start_time'):
                    elapsed = current_time - self.training_start_time
                else:
                    elapsed = 0

                loss_entry = {
                    "time": elapsed,
                    "loss": loss,
                    "step": self.training_status.get("current_step", 0),
                    "epoch": self.training_status.get("current_epoch", 0)
                }
                self.training_status["loss_history"].append(loss_entry)

                # Keep only last 1000 points to prevent memory issues
                if len(self.training_status["loss_history"]) > 1000:
                    self.training_status["loss_history"] = self.training_status["loss_history"][-1000:]

        elif "lr" in line_lower or "learning_rate" in line_lower:
            import re
            lr_match = re.search(r'lr[:\s]*([0-9.e+-]+)', line_lower)
            if lr_match:
                lr = float(lr_match.group(1))
                print(f"[{time.strftime('%H:%M:%S')}] Learning Rate: {lr:.2e}")

        elif any(keyword in line_lower for keyword in ["saving", "checkpoint", "validation", "gradient"]):
            print(f"INFO: [{time.strftime('%H:%M:%S')}] {line}")

        elif any(keyword in line_lower for keyword in ["error", "exception", "warning", "failed"]):
            print(f"[{time.strftime('%H:%M:%S')}] {line}")


    def _print_status_update(self, elapsed_time: float, last_step: int, last_loss: Optional[float], checkpoint_count: int):
        print(f"[{time.strftime('%H:%M:%S')}] Status Update:")
        print(f"- Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        print(f"- Progress: {self.training_status['progress']}%")
        print(f"- Current epoch: {self.training_status['current_epoch']}")
        print(f"- Checkpoints: {checkpoint_count}")
        if last_loss:
            print(f"- Last loss: {last_loss:.4f}")
        print("-"*40)

    def get_status(self) -> Dict[str, Any]:
        status = self.training_status.copy()
        return status

    def _auto_unwrap_final_checkpoint(self, save_dir: str):
        try:
            import subprocess
            from pathlib import Path

            checkpoint_files = list(Path(save_dir).glob("*.ckpt"))
            if not checkpoint_files:
                print(f"No checkpoint files found in {save_dir}")
                return

            latest_checkpoint = max(
                checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"Found latest checkpoint: {latest_checkpoint.name}")

            config = get_config()
            base_model = self.config.get(
                "baseModel", "stable-audio-open-small")
            base_model_configs = get_base_model_configs()

            if base_model not in base_model_configs:
                print(f"Unknown base model: {base_model}")
                return

            model_config_path = base_model_configs[base_model]["config"]

            unwrapped_dir = Path(save_dir) / "unwrapped"
            unwrapped_dir.mkdir(exist_ok=True)

            output_name = f"{latest_checkpoint.stem}_unwrapped"

            cmd = [
                sys.executable,
                str(config.get_path("stable_audio_tools") / "unwrap_model.py"),
                "--model-config", str(model_config_path),
                "--ckpt-path", str(latest_checkpoint),
                "--name", output_name,
                "--use-safetensors"
            ]

            print(f"Running unwrap command...")
            result = subprocess.run(cmd, cwd=str(config.get_path("stable_audio_tools")),
                                    capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Successfully unwrapped checkpoint")

                try:
                    latest_checkpoint.unlink()
                    print(f"Deleted wrapped checkpoint: {latest_checkpoint.name}")
                except Exception as e:
                    print(f"Could not delete wrapped checkpoint: {e}")

            else:
                print(f"Failed to unwrap checkpoint: {result.stderr}")

        except Exception as e:
            print(f"Auto-unwrap error: {e}")

    def stop_training(self) -> Dict[str, Any]:
        stopped_processes = []

        if self.training_process and self.training_process.poll() is None:
            try:
                self.training_process.terminate()
                try:
                    self.training_process.wait(timeout=10)
                    stopped_processes.append(
                        f"Process {self.training_process.pid}")
                except subprocess.TimeoutExpired:
                    self.training_process.kill()
                    stopped_processes.append(
                        f"Process {self.training_process.pid} (force killed)")
            except Exception as e:
                print(f"Error stopping stored process: {e}")

        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (proc.info['cmdline'] and
                        any('train.py' in cmd for cmd in proc.info['cmdline']) and
                            any('stable-audio' in cmd or 'models/fine_tuned' in cmd for cmd in proc.info['cmdline'])):

                        print(f"Found orphaned training process: PID {proc.info['pid']}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                            stopped_processes.append(
                                f"Orphaned process {proc.info['pid']}")
                        except psutil.TimeoutExpired:
                            proc.kill()
                            stopped_processes.append(
                                f"Orphaned process {proc.info['pid']} (force killed)")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            print("psutil not available, trying manual process search...")
            try:
                result = subprocess.run(
                    ['ps', 'aux'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'train.py' in line and ('stable-audio' in line or 'models/fine_tuned' in line):
                        pid = int(line.split()[1])
                        print(f"Found training process via ps: PID {pid}")
                        subprocess.run(['kill', '-TERM', str(pid)])
                        stopped_processes.append(f"Process {pid} (via ps)")
            except Exception as e:
                print(f"Error with fallback process search: {e}")

        self.training_status["is_training"] = False

        if stopped_processes:
            return {"status": "stopped", "message": f"Training stopped: {', '.join(stopped_processes)}"}
        else:
            return {"error": "No training process found to stop"}


_trainer_instance = None


def get_trainer() -> Optional[LoRATrainer]:
    return _trainer_instance


def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    global _trainer_instance
    if _trainer_instance and _trainer_instance.training_status["is_training"]:
        return {"error": "Training already in progress"}

    if not _trainer_instance:
        _trainer_instance = LoRATrainer(config)
    else:
        _trainer_instance.config = config
        _trainer_instance.training_status.update({
            "is_training": False,
            "error": None,
            "start_time": None,
            "estimated_completion": None
        })

    return _trainer_instance.start_training()


def get_training_status() -> Dict[str, Any]:
    global _trainer_instance
    if _trainer_instance:
        return _trainer_instance.get_status()
    else:
        return {
            "is_training": False,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "loss_history": [],
            "error": None,
            "start_time": None,
            "estimated_completion": None,
            "model_name": None,
            "current_step": 0,
            "checkpoints_saved": 0,
            "device_info": None
        }


def stop_training() -> Dict[str, Any]:
    global _trainer_instance
    if _trainer_instance:
        return _trainer_instance.stop_training()
    else:
        return {"error": "No training to stop"}
