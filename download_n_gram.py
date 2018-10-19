from google_ngram_downloader import readline_google_store
# https://github.com/dimazest/google-ngram-downloader

files = readline_google_store(ngram_len=5)

f_bundle = next(files, None)

sink = open("output.txt", "w")

while f_bundle is not None:
    fname, url, records = f_bundle
    print(fname)

    r = next(records, None)
    text = ""
    count = 0
    while r is not None:
        cText = r.ngram
        cCount = r.match_count
        cYear = r.year

        if cText != text:
            if count > 0:
                sink.write("{}\t{}\n".format(text, count))
            count = 0
            text = cText
        
        if cYear < 1980:
            # print(cText, cCount, cYear)
            count += cCount
        r = next(records, None)

    f_bundle = next(files, None)
