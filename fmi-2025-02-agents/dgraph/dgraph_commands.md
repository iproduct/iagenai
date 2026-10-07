mkdir ~/Desktop/dgraph-quickstart
cd ~/Desktop/dgraph-quickstart
``
## Step 2: Start Dgraph

From the directory you created, run Dgraph using the official Docker image:

```shell
docker run --detach --name dgraph-play `
  -v .:/dgraph `
  -p "8080:8080" `
  -p "9080:9080" `
  --pull always dgraph/standalone:latest
```

```shell
curl http://localhost:8080/health
```

```shell
docker exec -it dgraph-play dgraph live `
  -f 1million.rdf.gz `
  -s 1million.schema
```

```shell
docker run --rm -it -p 8060:8000 dgraph/ratel:latest
```

Query:
```dql
{
  film(func: has(genre), first: 3) {
    name@*
    genre { 
      name: name@. 
    }
    starring {
      performance.actor {
        name: name@.
      }
      performance.character {
        name: name@.
      }
    }
  }
}
```
