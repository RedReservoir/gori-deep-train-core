rm -rf api
mkdir api

sphinx-apidoc \
  -o api \
  ../src/gorideep \
  --force \
  --separate \
  --module-first
