
<h1>EF4-DASHBOARD</h1>  
<nbsp></nbsp>

<h3>Add your email address to the ef4_pull_dash.py file in two places</h3>

<b>Line 14:</b>  set_identity("email@email.com")
<nbsp></nbsp>
<b>Line 122:</b>  headers={"User-Agent": "email@email.com"},

<h3>Running the application.</h3>

python ef4_pull_dash.py

<h3>Viewing the application.</h3>

http://127.0.0.1:5000

<h3>Thoughts</h3>  
This data pull and chart is interesting. At first, I kind of felt intrusive since insider stock sales are linked to specific people and have a negative feeling attached when I think of the words or the idea. Realistically, insider stock sales are common. So common that I think they are seen as market neutral most of the time. Unless the volume of stock sales was really large. Then it may be a bearish signal. Insider stock purchases, however, I think are rare enough that they can be seen as possible bullish. If I already had stock positions in specific companies and wanted to keep track of insider selling, I may use this chart for it. However, I think my primary use for this chart would be to look for bullish signals. I created a separate faster app that finds companies with insider stock purchase activity that provides me with a ticker to use in this app as a second step. Edgar data pulls can be slow, so having specific tickers that have known insider stock purchase activity works better. The odds of randomly entering a ticker in this app and finding insider stock purchase activity are very low. MS CoPilot and Open Ai Codex were very helpful with this app since I am just getting started with Python and Edgar.  
