 #!/bin/bash
set -x
pip install omegaconf
pip install build/tensorrt_llm-0.19.0-cp312-cp312-linux_x86_64.whl

mv  /home/rfejgin/.local/lib/python3.12/site-packages/tensorrt_llm  /home/rfejgin/.local/lib/python3.12/site-packages/tensorrt_llm_bak
ln -s /code/tensorrt_llm/tensorrt_llm /home/rfejgin/.local/lib/python3.12/site-packages/tensorrt_llm
