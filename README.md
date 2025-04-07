

### Generation

```bash
export gpus=1
export model='bigcode/starcoder2-3b'
export crossfile_max_tokens=4096
export language=python
export task=test_intent_1891
export output_dir=./tmp/crosscodeeval_testrun/
python scripts/vllm_inference.py \
  --tp $gpus \
  --task $task \
  --language $language \
  --model $model \
  --output_dir $output_dir \
  --use_crossfile_context \
  --crossfile_max_tokens  $crossfile_max_tokens
```



### Metrics Calculation
After obtaining the generation, we can calculate the final metrics
```bash
export language=python
export ts_lib=./build/${language}-lang-parser.so; 
export task=test_intent_1891
export prompt_file=./data/${language}/${task}.jsonl 
export output_dir=./tmp/crosscodeeval_testrun/;  
python scripts/eval.py \
  --prompt_file $prompt_file \
  --output_dir $output_dir \
  --ts_lib $ts_lib \
  --language $language \
  --only_compute_metric
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
