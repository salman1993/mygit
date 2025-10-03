## Building a simplified git in Python


### Reading

- https://benhoyt.com/writings/pygit/
- https://blog.meain.io/2023/what-is-in-dot-git/
- https://wyag.thb.lt/#getting-started

### Setup

```bash
uv sync
```

### Testing

```bash
source .venv/bin/activate

mkdir test_dir && cd test_dir

mygit init
echo "hello world" > test.txt
echo "日本語" > nihongo.txt

mygit hash-object -w test.txt
mygit cat-file -p 3b18e512dba79e4c8300dd08aeb37f8e728b8dad

mygit status
mygit add test.txt
mygit status

tree .git

mygit commit -m "added test.txt"
mygit status
```
