"""
```shell
pip install huggingface-hub
```

#NOTE: Make sure you specify the name
       of the gguf file that you want to download
       , otherwise, it will download all of them! 

```shell 
huggingface-cli download \
  TheBloke/MistralLite-7B-GGUF \
  mistrallite.Q4_K_M.gguf \
  --local-dir downloads \
  --local-dir-use-symlinks False
```

#create a Modelfile with the following contents:
```shell
FROM ./downloads/mistrallite.Q4_K_M.gguf
```

#We then build an Ollama model using the following command:
```shell
ollama create mistrallite -f Modelfile
````

#verify with test
```shell
allama run mistrallite "What is Grafana?"
```
"""
