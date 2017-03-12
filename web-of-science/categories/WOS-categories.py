from bs4 import BeautifulSoup as Soup

soup = Soup(open("Web-of-Science-Cats.html"))

for x in soup.findAll("table"):
	if x["class"] == ["content"]:

		with open("WoS-categories", "w") as out:
			for y in x.findAll("p"):
				out.write(y.text + "\n") 
