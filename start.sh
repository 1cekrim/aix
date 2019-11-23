docker build -t aix "$PWD"
docker run -d -p 4000:4000 --name aix -v "$PWD":/srv/jekyll aix