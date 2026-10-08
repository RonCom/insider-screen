import duckdb

con = duckdb.connect("data/insider.duckdb", read_only=True)
q = lambda sql: print(con.execute(sql).df().to_string(index=False), "\n")

# candidates per year, and how many mention options or credit FINRA / the detection center
q("""SELECT year(release_date) AS year, count(*) AS releases,
            sum(is_insider_candidate::INT) AS candidates,
            sum((is_insider_candidate AND mentions_options)::INT) AS cand_options,
            sum((is_insider_candidate AND mentions_finra)::INT) AS cand_finra,
            sum((is_insider_candidate AND mentions_detection_center)::INT) AS cand_adc
     FROM raw.sec_litigation_releases GROUP BY 1 ORDER BY 1""")

# possible misses: non-candidates that talk about tipping or trading ahead of news
q("""SELECT lr_no, release_date, left(respondents, 60) AS respondents
     FROM raw.sec_litigation_releases
     WHERE NOT is_insider_candidate
       AND regexp_matches(text, '(?i)tipp(ed|ee|er)|ahead of (the|an) .{0,40}announce|confidential information.{0,80}(bought|purchased|sold)')
     ORDER BY lr_no""")

# possible false positives: candidates that never mention buying, selling or trading securities
q("""SELECT lr_no, release_date, left(respondents, 60) AS respondents
     FROM raw.sec_litigation_releases
     WHERE is_insider_candidate
       AND NOT regexp_matches(text, '(?i)purchas|bought|sold|traded|trading in')
     ORDER BY lr_no""")