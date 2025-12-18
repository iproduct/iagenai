PowerShell - monitoring Ollama server.log:
```powershell
 C:\Users\office27\AppData\Local\Ollama> Get-Content server.log -Wait -Tail 1000
 ```

```cmd 
curl http://localhost:11434/api/embeddings -d '{
  "model": "llama3.2",
  "prompt": "Here is an article about llamas..."
}'
curl "http://localhost:11434/api/embeddings" -d "{\"model\": \"llama3.2\", \"prompt\": \"Here is an article about llamas...\"}"
curl "http://localhost:11434/api/chat" -d "{\"model\": \"llama3.2\",  \"messages\": [{ \"role\": \"user\", \"content\": \"why is the sky blue?\"}]}"
netsh interface portproxy set v4tov4 listenport=11434 listenaddress=127.0.0.1 connectport=11434 connectaddress=0.0.0.0```