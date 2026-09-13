from brownlow import afl_reports

MATCH_HTML = """
<div data-ui-tab="Match Report" class="mc-tabs__content js-mc-tab">
  <script data-article-structure data-article-id="1589873" type="application/ld+json"></script>
</div>
<div data-ui-tab="Team Stats" class="mc-tabs__content js-mc-tab"></div>
"""

ARTICLE_HTML = """
<h1 class="article__heading">Suns spoil veteran&#39;s 300th with tough win over Saints</h1>
<div class="article__body">
  <p>Short</p>
  <p>Gold Coast beat St Kilda 14.19 (103) to 12.8 (80) at Marvel Stadium on Thursday evening.</p>
  <p>More detail that should not be included in the excerpt.</p>
</div>
"""


def test_match_page_article_id():
    assert afl_reports.match_page_article_id(MATCH_HTML) == "1589873"
    assert afl_reports.match_page_article_id("<html></html>") is None


def test_article_summary_extracts_headline_and_lead():
    headline, text = afl_reports.article_summary(ARTICLE_HTML)
    assert headline.startswith("Suns spoil veteran's")
    assert text.startswith("Gold Coast beat St Kilda")


def test_article_summary_truncates_at_sentence_boundary():
    sentence = "The midfield battle swung on a late goal. " * 60
    html = f'<div class="article__body"><p>{sentence}</p></div>'
    _, text = afl_reports.article_summary(html)
    assert len(text) <= afl_reports.MAX_SUMMARY_CHARS + 3
    assert text.endswith(".")


def test_article_summary_combines_leading_paragraphs():
    html = (
        '<div class="article__body">'
        "<p>Gold Coast beat St Kilda 14.19 (103) to 12.8 (80) at Marvel Stadium on Thursday evening, "
        "closing the season with four points that meant little in context.</p>"
        "<p>Max Hall was the standout, gathering 37 disposals and driving the Saints forward all night.</p>"
        "<p>A third paragraph that is long enough to pass the sixty character filter for inclusion.</p>"
        "</div>"
    )
    _, text = afl_reports.article_summary(html)
    assert "Max Hall" in text
    assert text.count("\n\n") >= 1
