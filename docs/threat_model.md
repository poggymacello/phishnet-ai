# Threat model

This covers two different things that are easy to conflate: how a
phishing URL's tactics map to MITRE ATT&CK, and the threat model *of the
detector itself* -- how it could be evaded, and where it sits in a real
mail/browser security pipeline. Everything here is defensive analysis.
There is no evasion proof-of-concept, homograph generator, or cloaking
tooling anywhere in this project.

## Phishing tactic to MITRE ATT&CK mapping

| Tactic | ATT&CK technique | URL-level signal |
|---|---|---|
| Credential harvesting via a fake login page | T1566.002 Phishing: Spearphishing Link | Long, subdomain-heavy URL mimicking a brand name; often no real HTTPS certificate for the actual brand domain |
| Homograph / lookalike domain (e.g. `paypa1.com`, IDN punycode tricks) | T1583.001 Acquire Infrastructure: Domains (in service of T1566.002) | High digit/letter-substitution ratio in the domain; punycode (`xn--`) prefix; visually similar TLD |
| URL shortener / redirector to hide the real destination | T1027.302 Obfuscated Files or Information: Steganography-adjacent URL obfuscation (closest fit; ATT&CK doesn't have a URL-shortener-specific ID) | Short domain, no path, from a small set of known shortener services; final destination only visible after a redirect this project deliberately never follows server-side |
| Free/automated TLS certificate abuse (Let's Encrypt phishing pages) | T1588.004 Obtain Capabilities: Digital Certificates | `IsHTTPS=1` no longer distinguishes legitimate from phishing the way it did a few years ago -- see Limitations below, this is exactly the trend that erodes the `IsHTTPS` signal this dataset over-relies on |
| Cloaking (serving a benign page to crawlers/security scanners, the real phishing page to a targeted victim) | T1027.006 Obfuscated Files or Information: HTML Smuggling (closest fit) | Invisible to a URL-only, pre-fetch model by construction; would require repeated fetches from different vantage points, which this project deliberately does not do (see Deployment section) |

This mapping connects a URL-level pattern to what an analyst would
actually flag it as, not a claim that the model reasons about attacker
tactics -- it classifies character n-grams and structured URL features,
nothing more.

## Evasion: how an attacker could evade this specific detector

Discussion only, no working evasion code exists in this repository.

- **The `IsHTTPS` shortcut is real and will degrade.** The leakage
  investigation (README's Leakage Controls) found that 100% of PhiUSIIL's
  legitimate URLs use HTTPS, and the depth-3 decision tree's root split is
  `IsHTTPS`. An attacker using a free automated certificate (Let's
  Encrypt, and equivalents) on a phishing domain defeats exactly the
  single strongest feature this dataset offers, for free, today. This is
  not a hypothetical: free-cert phishing has been common practice for
  years. Any deployment relying on this model's current feature
  importances should assume this specific shortcut degrades over time,
  not treat today's numbers as durable.
- **Character n-gram mimicry.** The TF-IDF baseline scores based on
  substrings like `http:` vs `https:`, brand-name tokens, and TLD
  patterns. An attacker who registers a domain avoiding the n-grams this
  particular model learned to associate with phishing (while still being
  a phishing domain) is not meaningfully harder to evade than any other
  bag-of-n-grams text classifier -- it has no semantic understanding of
  what makes a domain suspicious beyond the training distribution's
  surface statistics.
- **New/unseen legitimate domains look statistically unusual too.** The
  eTLD+1 group split specifically tests this: the model has never seen a
  brand-new domain's other URLs during training. A newly registered
  legitimate small-business domain (long name, few established signals)
  can resemble the URL-level statistics of a phishing domain more than an
  established brand does -- this is a genuine false-positive risk, not
  just an evasion path, and matters for real deployment fairness (new
  legitimate sites shouldn't be systematically penalized).
- **Feature-blind spots.** The deployed model only ever sees the raw URL
  string. It has no access to page content, WHOIS/registration age,
  hosting infrastructure, or DNS history -- all of which are real signals
  production phishing detectors use and this project deliberately doesn't
  have (see Deployment section for why: fetching the page server-side is
  an SSRF vector this project avoids by design).

**Mitigation, conceptually:** none of the above is solved by this
project. A real deployment would combine URL-level scoring (what this
project does) with domain-age/WHOIS signals, DNS reputation feeds,
and browser-side rendering checks in an isolated sandbox (not
server-side), plus periodic retraining as attacker infrastructure shifts
away from whatever features are currently strongest.

## Where this sits in a real pipeline, and why the operating point matters

A URL-only phishing classifier like this one is typically positioned as
a pre-click filter: a browser extension, an email gateway link-rewriter,
or a chat/SMS link scanner, running before a user ever reaches the page.

```
inbound URL (email link, chat message, typed address)
    -> [this classifier, URL-only, no page fetch]
    -> block / warn / allow decision
    -> (optionally) sandboxed page render + deeper analysis for borderline scores
```

Recall at a fixed, low false-positive rate (reported at 1% and 5% FPR
budgets in the README) is the metric that actually matters here: a
pre-click filter that blocks legitimate links too often trains users to
click through warnings, which defeats the entire point of having one. A
high-recall, high-false-positive model is not obviously better than a
more conservative one in this specific deployment context.
