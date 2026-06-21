"""ufcstats.com scraper: Scrapy spider + BeautifulSoup parsers.

Parsing logic lives in `parsers.py` as pure functions (HTML string -> Pydantic
models) so it is unit-testable offline against saved fixtures. The Scrapy spider,
challenge-solving middleware, and DuckDB pipeline wire those parsers to the live
site and storage.
"""
