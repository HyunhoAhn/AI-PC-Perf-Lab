# Tools

## Runtime Verification

Verify ONNX Runtime and providers in the active environment:

```powershell
python -c "import onnxruntime as ort; print('onnxruntime=' + ort.__version__); print('providers=' + ','.join(ort.get_available_providers()))"
```

## Tools in ./tools 

*Notes: $runID is unique folder name for storing your results at ../results/raw


### Capture environment:

```powershell
python tools/capture_env.py --run-id $runId
```
It will generate `results/raw/<run_id>/env_history.jsonl` with details on your environments information. It ensures that the environment details are captured and versioned for each run, allowing you to track changes over time and correlate them with performance results.


### Capture command execution:

Example with ONNX Runtime info, but you can replace the command with your benchmark or profiling command. The tool will capture stdout, stderr, and metadata for the run.
```powershell
python tools/run_capture.py --run-id $runId -- python -c "import onnxruntime as ort; print('onnxruntime_version=' + ort.__version__); print('providers=' + ','.join(ort.get_available_providers()))"
```

It will generate the following files:
   - `results/raw/<run_id>/stdout.log`
   - `results/raw/<run_id>/stderr.log`
   - `results/raw/<run_id>/metadata.json`   

The `metadata.json` will include the command, start/end timestamps, and environment metadata reference.
The stdout and stderr logs will capture the respective outputs of the command execution.