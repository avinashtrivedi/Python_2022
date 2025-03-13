from fpdf import FPDF
import os

#create an instance of fpdf
pdf = FPDF ()
pdf.set_auto_page_break(0)
pdf.set_compression(compress: bool)

img_list = ['sixbit_bid_ask.png','sixbit_turnover.png','xetrabit_bid_ask.png','xetrabit_turnover.png',
			'parisbit_bid_ask.png','parisbit_turnover.png','amsbit_bid_ask.png','amsbit_turnover.png',
			'sixeth_bid_ask.png','sixeth_turnover.png','xetraeth_bid_ask.png','xetrabit_turnover.png',
			'pariseth_bid_ask.png','pariseth_turnover.png','amseth_bid_ask.png','amseth_turnover.png',
			'sixbaskets_bid_ask.png','sixbaskets_turnover.png','xetrabaskets_bid_ask.png','xetrabaskets_turnover.png',
			'parisbaskets_bid_ask.png','parisbaskets_turnover.png','amsbaskets_bid_ask.png','amsbaskets_turnover.png',
			'sixdot_bid_ask.png','sixdot_turnover.png','xetradot_bid_ask.png','xetradot_turnover.png',
			'parisdot_bid_ask.png','parisdot_turnover.png','amsdot_bid_ask.png','amsdot_turnover.png',
			'sixsol_bid_ask.png','sixsol_turnover.png','xetrasol_bid_ask.png','xetrasol_turnover.png',
			'parissol_bid_ask.png','parissol_turnover.png','amssol_bid_ask.png','amssol_turnover.png']

#add new pages with the image
for img in img_list:
	pdf.add_page()
	pdf.image(img)

#save the output file
pdf.output("ProductComparison.pdf")

print("Adding all your images into a pdf file")
print("Images pdf is created and saved into the following path folder :\n", os.getcwd())

