
import unittest
import timing 


class TestTimingProgram(unittest.TestCase):


	def test_makeTimingDict(self):

		ans = timing.makeTimingDict("ultra.tiny.title.basics.tsv")
		soln = {1713499: {'primaryTitle': 'Meena', 'titleType': 'movie', 'runtimeMinutes': 0, 'startYear': '2012'}, 2543164: {'primaryTitle': 'Arrival', 'titleType': 'movie', 'runtimeMinutes': 116, 'startYear': '2016'}}
		self.assertEqual(ans, soln)


		ans2 = timing.makeTimingDict("tiny.title.basics.tsv")
		soln2 = {11351144: {'primaryTitle': 'Kapitel 249', 'titleType': 'tvEpisode', 'runtimeMinutes': 0, 'startYear': '2010'}, 13973364: {'primaryTitle': 'Carlos Miguel', 'titleType': 'tvEpisode', 'runtimeMinutes': 0, 'startYear': '2006'}, 4201304: {'primaryTitle': 'Episode #1.9', 'titleType': 'tvEpisode', 'runtimeMinutes': 52, 'startYear': '2004'}, 5991910: {'primaryTitle': 'Episode #8.54', 'titleType': 'tvEpisode', 'runtimeMinutes': 0, 'startYear': '2016'}, 7052686: {'primaryTitle': 'Episode #1.172', 'titleType': 'tvEpisode', 'runtimeMinutes': 0, 'startYear': '1995'}, 3693356: {'primaryTitle': 'Tyven - tyven', 'titleType': 'tvEpisode', 'runtimeMinutes': 27, 'startYear': '2005'}, 54135: {'primaryTitle': "Ocean's 11", 'titleType': 'movie', 'runtimeMinutes': 127, 'startYear': '1960'}, 240772: {'primaryTitle': "Ocean's Eleven", 'titleType': 'movie', 'runtimeMinutes': 116, 'startYear': '2001'}, 450069: {'primaryTitle': 'The Oceans 11 Story', 'titleType': 'video', 'runtimeMinutes': 70, 'startYear': '2001'}, 9832806: {'primaryTitle': 'Blue Mountains', 'titleType': 'tvEpisode', 'runtimeMinutes': 23, 'startYear': '2019'}}
		self.assertEqual(ans2, soln2)




if __name__ == '__main__':
	unittest.main()