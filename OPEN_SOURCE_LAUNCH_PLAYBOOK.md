# Open-Source AI Framework Launch Playbook
## How to Go Viral and Gain Thousands of GitHub Stars

Research compiled from analysis of: FastAPI (80k+ stars), uv (50k+ stars), CrewAI (25k+ stars), LangChain (100k+ stars), AutoGPT (170k+ stars), Agno/Phidata (18k+ stars), Pydantic-AI (8k+ stars), smolagents (15k+ stars), LiteLLM (17k+ stars), and dozens of articles on open-source growth.

---

## Table of Contents

1. [README Patterns That Win](#1-readme-patterns-that-win)
2. [Repository Structure and Polish](#2-repository-structure-and-polish)
3. [Documentation Strategy](#3-documentation-strategy)
4. [Demo and Visual Strategy](#4-demo-and-visual-strategy)
5. [Launch Strategy: The First 48 Hours](#5-launch-strategy-the-first-48-hours)
6. [Platform-Specific Launch Tactics](#6-platform-specific-launch-tactics)
7. [Community Building](#7-community-building)
8. [Sustained Growth Tactics](#8-sustained-growth-tactics)
9. [What Makes AI/Agent Repos Specifically Go Viral](#9-what-makes-aiagent-repos-specifically-go-viral)
10. [Anti-Patterns to Avoid](#10-anti-patterns-to-avoid)
11. [Pre-Launch Checklist](#11-pre-launch-checklist)
12. [Post-Launch Week-by-Week Plan](#12-post-launch-week-by-week-plan)

---

## 1. README Patterns That Win

### Analysis of Top Repos

**FastAPI (80k+ stars) - The Gold Standard README**

FastAPI's README is the single best template in the Python ecosystem. Structure:

1. **Centered logo** - Professional, branded SVG logo
2. **Tagline in italics** - "FastAPI framework, high performance, easy to learn, fast to code, ready for production"
3. **Badge row** - Test status, coverage, PyPI version, Python versions
4. **Documentation + Source Code links** - Immediately below badges, separated by horizontal rules
5. **One-sentence description** - Clear positioning statement
6. **Key Features as bullet list** - Each bold with a brief explanation. Uses quantified claims ("200% to 300% faster to code", "40% fewer bugs")
7. **Sponsors section** - Social proof through logos
8. **Opinions/Testimonials** - Quotes from Microsoft, Uber, Netflix, Cisco, spaCy creators. EACH with a verifiable reference link
9. **Minimal install** - Single `pip install` command
10. **Complete working example** - Copy-paste-run code in <20 lines
11. **"Check it" section** - Shows what happens when you run it (screenshots of Swagger UI)
12. **Progressive complexity** - Starts simple, then shows an "Example upgrade"
13. **Performance benchmarks** - Links to independent TechEmpower benchmarks
14. **Dependencies listed** - "Stands on the shoulders of giants" (credits Starlette, Pydantic)

**Key insight from FastAPI**: The README SELLS the project in the first screenful. You should be convinced to try it within 10 seconds of landing on the repo.

**uv (50k+ stars)**

1. **Hero image/animation** - Terminal GIF showing 10-100x speed vs pip
2. **One-line description** - "An extremely fast Python package and project manager, written in Rust."
3. **Highlights as bullets** - Speed comparison, drop-in pip replacement, single binary
4. **Installation** - One-liner (`curl` or `pip install`)
5. **Getting started** - Minimal commands
6. **Benchmark charts** - Speed comparisons front-and-center
7. **Feature list** - Comprehensive but scannable

**Key insight from uv**: Lead with a DRAMATIC visual demonstration of the core value prop. The speed GIF IS the README.

**CrewAI (25k+ stars)**

1. **Logo + tagline** - "Framework for orchestrating role-playing, autonomous AI agents"
2. **Badges** - Stars, PyPI, Discord members, contributors
3. **Why CrewAI?** - Positioned against the "why not just use LangChain" question
4. **Quick start code** - Shows the "crew" metaphor with agents having roles
5. **Key concepts** - Agents, Tasks, Crews explained simply
6. **Examples gallery** - Links to many use-case examples

**Key insight from CrewAI**: The metaphor IS the product. "Crew of AI agents" is instantly understandable and tweetable. The framework sold itself through its naming/mental model.

**Agno/Phidata (18k+ stars)**

1. **Performance benchmarks front-and-center** - "1000x faster agent instantiation than LangGraph"
2. **Comparison tables** - Direct benchmarks vs competitors
3. **Multi-modal support callout** - Shows breadth
4. **Code example** - Simple agent creation

**Key insight from Agno**: Bold, quantified competitive claims drive virality. "1000x faster" is shareable.

**Pydantic-AI (8k+ stars)**

1. **Leveraged the Pydantic brand** - Instant credibility from an established project
2. **Type-safety positioning** - "Agent Framework / shim to use Pydantic with LLMs"
3. **Model-agnostic** - Support for OpenAI, Anthropic, Gemini, etc.
4. **Dependency injection** - Novel technical differentiator

**Key insight from Pydantic-AI**: Credibility transfer from an existing successful project is incredibly powerful.

### The Ideal README Structure

```
1. Centered logo (SVG, professional)
2. One-line tagline (italicized, under logo)
3. Badge row (build, coverage, PyPI, Python versions, license, Discord)
4. Horizontal rule
5. Documentation link | Source Code link
6. Horizontal rule
7. One-paragraph description (2-3 sentences max)
8. Key Features (4-6 bullet points, bold headers with brief text)
   - Lead with the DIFFERENTIATOR
   - Include at least one quantified claim
9. Quick Install (one command)
10. Minimal Example (< 20 lines, copy-paste-runnable)
11. "What happens" section (screenshot or output)
12. More Examples (progressive complexity, or link to examples/)
13. Benchmarks / Comparisons (if you have them)
14. Testimonials / Social Proof (quotes from known devs/companies)
15. Documentation link (again)
16. Community links (Discord, Twitter)
17. Contributing section (or link to CONTRIBUTING.md)
18. License
```

### README Writing Rules

- **First screenful decides everything.** A developer who lands on your GitHub page will decide to star or leave within 5-10 seconds. The top of the README must answer: "What is this? Why should I care? How do I try it?"
- **Show, don't tell.** Code > prose. GIFs > screenshots > text.
- **Quantify everything.** "Fast" means nothing. "10x faster than X" means everything.
- **Name competitors.** "Unlike LangChain/CrewAI..." positions you and saves the reader research.
- **Make the install command visible within 1-2 scrolls.** If someone has to scroll 5 screens to find `pip install`, you've lost them.
- **The example must work if copy-pasted.** Test this. Seriously. If someone installs your package and pastes the README example and it errors, you'll get negative word-of-mouth.

---

## 2. Repository Structure and Polish

### Essential Files

```
repo/
  README.md               # The sales page
  LICENSE                  # MIT or Apache 2.0 (MIT is more permissive/popular)
  CONTRIBUTING.md          # How to contribute
  CODE_OF_CONDUCT.md       # Standard covenant
  CHANGELOG.md             # Shows the project is alive
  pyproject.toml           # Modern Python packaging
  .github/
    ISSUE_TEMPLATE/
      bug_report.md
      feature_request.md
    PULL_REQUEST_TEMPLATE.md
    workflows/
      test.yml             # CI that runs on every PR
      release.yml           # Automated PyPI publishing
    FUNDING.yml             # GitHub Sponsors / Open Collective
  docs/                    # Full documentation (MkDocs Material or similar)
  examples/                # Runnable examples for common use cases
  tests/                   # Comprehensive test suite
  src/yourpackage/         # The actual code (src layout)
```

### Badges That Matter

In order of importance:
1. **Build/CI status** - Shows it works
2. **PyPI version** - Shows it's installable
3. **Python versions** - Shows compatibility
4. **License** - MIT = most welcoming
5. **Coverage** - Shows quality (only if > 80%)
6. **Discord/community** - Shows it's alive
7. **Downloads** - Shows traction (only add once numbers are impressive)
8. **GitHub stars** - Shows social proof (only add once numbers are impressive)

### Topics / Tags

Set GitHub topics on your repository. These affect discoverability in GitHub search:
- `python`, `ai`, `agents`, `llm`, `machine-learning`, `ai-agents`, `openai`, `anthropic`, `framework`

### Repository Settings

- Enable GitHub Discussions (for Q&A and community)
- Set up a good repo description (this shows in search results)
- Add a website URL (link to your docs)
- Pin important issues for new contributors
- Use GitHub Releases with good release notes

---

## 3. Documentation Strategy

### What Successful AI Frameworks Do

**Tier 1 - Required for launch:**
- Getting Started / Quickstart guide (5 minutes to first result)
- Installation guide
- Core concepts (explain your mental model)
- At least 3-5 working examples
- API reference (auto-generated is fine)

**Tier 2 - Required within first month:**
- Tutorials (step-by-step, building something real)
- How-to guides (task-oriented: "How to use tool X with framework Y")
- Cookbook / Recipes
- FAQ

**Tier 3 - Growth phase:**
- Architecture / internals explanation
- Comparison with alternatives (be fair but clear about advantages)
- Migration guides (from LangChain, CrewAI, etc.)
- Video tutorials
- Blog posts explaining design decisions

### Documentation Tooling

The most successful Python projects use:
- **MkDocs + Material for MkDocs** (FastAPI, Pydantic, uv all use this)
- Hosted on GitHub Pages or Read the Docs
- Includes search, dark mode, navigation
- Code examples that are tested (via `pytest` or `doctest`)

### The "Time to Hello World" Metric

The single most important documentation metric is: **How long does it take from `pip install` to seeing a working result?**

- FastAPI: ~60 seconds (install, paste code, run, see Swagger UI)
- uv: ~30 seconds (install, `uv init`, `uv run`)
- CrewAI: ~2 minutes (install, define agents, see output)

Your target should be **under 2 minutes**. If it takes longer, simplify your quickstart.

---

## 4. Demo and Visual Strategy

### GIFs and Videos

**What works:**
1. **Terminal recording GIF** - Show the framework in action. Use `asciinema` or `vhs` (by Charm) to record terminal sessions. Convert to GIF.
2. **Side-by-side comparison** - Your framework vs. doing it the hard way (or vs. competitor). uv's "10-100x faster" GIF is the canonical example.
3. **Output showcase** - Show what the agent actually produces. If your agents can browse the web, show the web browsing. If they can write code, show the code.
4. **Architecture diagram** - A clean diagram showing how components fit together.

**Tools for creating demos:**
- `vhs` (Charm) - Record terminal sessions as GIFs with a script
- `asciinema` - Terminal recording (also renders in browser)
- `terminalizer` - Another terminal GIF tool
- Excalidraw - For architecture diagrams (hand-drawn style is trendy)
- Mermaid - For diagrams in markdown

### Demo Project / Showcase App

Build one compelling demo app that shows off the framework's best capabilities:
- It should solve a real problem people care about
- It should be visually interesting (not just text in/text out)
- It should be runnable with a single command
- Ideas: "Build a research assistant in 10 lines", "Create a code review agent", "Multi-agent debate system"

---

## 5. Launch Strategy: The First 48 Hours

### Pre-Launch (2-4 Weeks Before)

1. **Build in public** - Tweet/post about the development process. Share snippets. Ask for opinions on design decisions. This builds an audience before launch.
2. **Recruit early testers** - Find 10-20 developers to try the framework and give feedback. They become your first advocates.
3. **Prepare all materials:**
   - README polished and tested
   - Documentation site live
   - PyPI package published and installable
   - At least 5 working examples
   - Blog post written
   - Social media posts drafted
   - Hacker News post drafted
   - Demo GIF/video created
4. **Line up supporters** - DM people who might share. AI influencers on Twitter, Python community members, people who have complained about existing frameworks.
5. **Time it right:**
   - Launch on Tuesday-Thursday (Mon/Fri are worse)
   - Post to HN around 8-9 AM EST (US morning, EU afternoon)
   - Avoid major conferences, holidays, or big industry news days

### Launch Day Sequence

1. **Morning (8 AM EST):**
   - Publish the blog post
   - Post to Hacker News ("Show HN")
   - Tweet the announcement with GIF/demo
   - Post to Reddit (r/Python, r/MachineLearning)

2. **Immediately after:**
   - Share HN link with early supporters, ask them to upvote (subtly)
   - Post in relevant Discord servers (Python, AI, LLM communities)
   - Post on LinkedIn

3. **Throughout the day:**
   - Respond to EVERY comment on HN, Reddit, Twitter
   - Fix any bugs reported immediately (shows responsiveness)
   - Thank people who star the repo
   - Retweet/share positive feedback

4. **Day 2:**
   - Follow up on any discussions
   - Post a "Day 1 results" thread on Twitter (people love traction stories)
   - Submit to newsletters (Python Weekly, AI Weekly, etc.)
   - Post on Dev.to, Hashnode

### The Snowball Effect

The first 48 hours create a feedback loop:
- HN front page -> stars -> trending on GitHub -> more stars -> more coverage -> more stars
- Getting on GitHub Trending (github.com/trending) is critical. This requires a burst of stars (typically 50-100+ in a day).
- GitHub Trending drives organic discovery for days/weeks after the initial launch.

---

## 6. Platform-Specific Launch Tactics

### Hacker News ("Show HN")

**Format:**
```
Show HN: [Name] - [One-line description]
```

**Rules for success:**
- Title must be concise and descriptive
- The "Show HN" prefix is required for project showcases
- Post text should include: what it is, why you built it, what makes it different, and a link
- Do NOT ask for upvotes (HN detects and penalizes this)
- Respond to every comment, especially critical ones. HN respects founders who engage thoughtfully with criticism.
- Be honest about limitations. HN users WILL find them, and honesty earns respect.
- Technical depth wins. HN loves "here's how we solved X" more than marketing speak.

**Example HN post text:**
```
Show HN: FrameworkName - AI agents that actually work in production

Hi HN, I built FrameworkName because [personal motivation/pain point].

After using LangChain/CrewAI for [X], I found [specific problems].
FrameworkName solves this by [technical approach].

Key differences:
- [Differentiator 1 with quantified claim]
- [Differentiator 2]
- [Differentiator 3]

It's MIT licensed, pip-installable, and you can build a working agent in <20 lines.

GitHub: [link]
Docs: [link]
```

**What makes HN posts succeed:**
- Solve a real problem the poster personally experienced
- Technical substance, not marketing language
- Honesty about tradeoffs and limitations
- Being a single person or small team (HN roots for underdogs)
- Responding fast and thoughtfully in comments

### Reddit

**r/Python (~1.5M members)**
- Best for general Python tool announcements
- Use the "Show" or "Library" flair
- Include what problem it solves and a code example
- Follow up in comments with more details
- Do NOT be salesy. Redditors hate self-promotion that feels like ads.

**r/MachineLearning (~3M members)**
- More academic/research-oriented
- Frame it as a technical contribution
- Include benchmarks or novel approaches
- Flair: [Project]

**r/LocalLLaMA (~800k members)**
- Focused on running LLMs locally
- If your framework supports local models (Ollama, vLLM, etc.), this is goldmine
- Extremely engaged community, love practical tools
- Show it working with open-source models

**r/artificial (~800k members)**
- More mainstream AI audience
- Good for impressive demos

**r/LangChain / r/ChatGPT / similar niche subs**
- Good for reaching people already in the ecosystem
- Can be more promotional if you're solving their specific pain points

**Reddit Rules:**
- Post at ~10 AM EST for best visibility
- Never post the same content to multiple subs simultaneously (looks like spam)
- Stagger posts across 2-3 days
- Engage authentically in comments
- If posting in multiple subs, customize the content for each community

### Twitter/X

**Launch Thread Strategy:**
```
Tweet 1: The hook
"I just open-sourced [name]: [one-line pitch]

[GIF or screenshot]

Here's why I built it and how it works: (thread)"

Tweet 2: The problem
"The problem with current AI agent frameworks:
- [Pain point 1]
- [Pain point 2]
- [Pain point 3]"

Tweet 3: The solution
"[Name] solves this with [approach]:
[Code screenshot or snippet]"

Tweet 4-6: Feature highlights
Each tweet = one feature with a visual

Tweet 7: Social proof / benchmark
"In benchmarks, [name] is Nx faster than [competitor]"

Tweet 8: Call to action
"Star on GitHub: [link]
Install: pip install [name]
Docs: [link]

If this is useful, RT to help others find it."
```

**Twitter Tactics:**
- Tag relevant people (AI researchers, Python influencers, framework maintainers you admire)
- Use relevant hashtags sparingly (#Python #AI #LLM #OpenSource)
- Post the thread between 9-11 AM EST or 1-3 PM EST
- Pin the thread to your profile
- Quote-tweet with additional context throughout the day
- Engage with every reply

**Key AI/Python Twitter accounts to be aware of (potential amplifiers):**
- AI researchers and developers with large followings
- Developer advocacy accounts at OpenAI, Anthropic, Google
- Python community leaders
- "AI news" accounts that reshare cool projects

### Product Hunt

- Less critical for developer tools than HN/Reddit/Twitter
- But can drive a burst of stars if you reach top 5 of the day
- Best launched on Tuesday-Thursday
- Need a maker account and ideally a "hunter" (someone popular who submits it for you)
- Make a compelling tagline, gallery images, and first comment

### LinkedIn

- Surprisingly effective for B2B/enterprise-oriented AI tools
- Write a personal post about why you built it (story format)
- Tag your company, relevant connections
- Good for reaching engineering managers and CTOs who make tool decisions

### Dev.to / Hashnode / Medium

- Write a "Building X: Why and How" blog post
- Cross-post to all three
- These rank well in Google and drive long-term organic traffic
- Include lots of code examples

### Newsletters

Submit to:
- **Python Weekly** (pythonweekly.com) - Major Python newsletter
- **Pycoders Weekly** - Another major Python newsletter
- **The Batch** (deeplearning.ai) - AI newsletter by Andrew Ng
- **TLDR** (tldrnewsletter.com) - Huge tech newsletter
- **Console.dev** - Curated developer tools newsletter
- **AI Weekly** - AI-focused newsletter
- **Changelog** - Open source newsletter/podcast
- **Hacker Newsletter** - Weekly best of HN

---

## 7. Community Building

### Discord Server

- Create before launch
- Channels: #announcements, #general, #help, #showcase, #feature-requests, #contributing
- Be active daily (especially first month)
- The Discord member count becomes a badge of credibility
- Add a Discord invite link to README and docs

### GitHub Discussions

- Enable for Q&A, ideas, and show-and-tell
- Move support from Issues to Discussions
- Pin important threads

### Making Contributors Feel Welcome

- Label issues with "good first issue" and "help wanted"
- Write detailed CONTRIBUTING.md with setup instructions
- Respond to PRs quickly (< 24 hours)
- Thank contributors publicly
- Add contributors to README (use all-contributors bot)
- Create a Discord role for contributors

### Building an Ecosystem

- Create a `awesome-[framework]` repo with community projects
- Highlight community examples in your docs
- Offer to co-author blog posts with users
- Create integration guides for popular tools

---

## 8. Sustained Growth Tactics

### Content Strategy (Monthly)

1. **Blog posts** - "How we solved X", "Benchmarks update", "New feature deep-dive"
2. **Tutorials** - "Build a [specific thing] with [framework]"
3. **Comparison posts** - Fair comparisons with alternatives (drives search traffic)
4. **Release announcements** - Each release is a marketing event
5. **User stories** - How companies/developers use it

### SEO and Discoverability

- Good docs site with proper meta tags ranks in Google
- Blog posts with titles like "Best Python AI agent framework 2026" capture search intent
- README and docs should naturally include searchable keywords
- PyPI package description matters for `pip search` and PyPI web search

### Regular Releases

- Ship frequently (weekly or biweekly)
- Each release is a reason to post on social media
- Use semantic versioning
- Write clear changelogs
- Major versions are marketing events

### Conference Talks

- Submit to PyCon, EuroPython, local Python meetups
- Lightning talks are easiest to get accepted
- Record and post to YouTube
- Conference talks generate organic backlinks and credibility

### Integrations and Partnerships

- Build integrations with popular tools (LangSmith, Weights & Biases, etc.)
- Get listed in partner docs/marketplaces
- Each integration is a co-marketing opportunity

---

## 9. What Makes AI/Agent Repos Specifically Go Viral

### Pattern Analysis

After analyzing the viral trajectories of the top AI repos:

**AutoGPT (170k stars)** - Went viral because it was the first "autonomous agent" people could try. Tapped into GPT-4 launch hype. The DEMO was everything.

**LangChain (100k stars)** - First mover in LLM orchestration. Solved a real pain point (chaining LLM calls) when GPT-3.5 exploded. Prolific content marketing and fast iteration.

**CrewAI (25k stars)** - Perfect metaphor ("crew of agents"). Launched at peak AI agent hype. Simple enough for beginners. Great Twitter presence by the creator.

**uv (50k stars)** - "10-100x faster" is irresistible. Backed by Astral (credibility). The benchmark GIF went viral on its own.

**FastAPI (80k stars)** - Solved real problems with Flask/Django for APIs. Incredible documentation. Leveraged Python type hints trend. The README IS a masterclass.

### What They All Have in Common

1. **Clear, memorable positioning** - You can explain what each does in one sentence
2. **Rode a technology wave** - GPT-4, LLMs, Python typing, Rust performance
3. **Solved a specific pain point** - Not "yet another framework" but "this solves X that Y can't"
4. **Incredible first impression** - README and first 5 minutes of usage were polished
5. **Active creator presence** - The maintainers were visible on social media, responsive to issues
6. **Launched at the right moment** - Timing with market demand

### AI Framework-Specific Differentiators That Get Stars

The things that make people star AI agent repos specifically:

1. **"It actually works" factor** - Most AI agent frameworks feel like toys. If yours works reliably in production, shout about it.
2. **Simplicity** - "Build an agent in 5 lines" beats "configure 20 files to get started"
3. **Speed/Performance** - Benchmark against competitors
4. **Type safety / Developer experience** - Python devs increasingly expect IDE autocompletion
5. **Model agnosticism** - Supporting OpenAI + Anthropic + local models is expected
6. **Novel capability** - What can your agents do that others can't? (structured output, multi-agent, tool use, memory, etc.)
7. **Production-readiness signals** - Observability, error handling, retries, streaming

### The "FOMO Star"

Many GitHub stars come from people who:
- Saw it trending and star to "bookmark it"
- Want to look current by engaging with trending repos
- Follow AI Twitter and star everything that looks promising

This means **creating FOMO and social proof is just as important as technical merit** in the first few days.

---

## 10. Anti-Patterns to Avoid

1. **Star-begging** - Never put "Please star this repo" at the top of your README. It looks desperate. (Small footer mentions are OK.)
2. **Overclaiming** - Don't claim "10x better than everything" without benchmarks. HN will destroy you.
3. **Vaporware README** - Don't make the README describe features that don't work yet. People will install, find broken things, and leave negative comments.
4. **No working example** - If someone can't get a working result in 5 minutes, they move on.
5. **Ignoring issues** - Unresponded issues signal an abandoned project. Respond within 24 hours.
6. **Launching too early** - Wait until the quickstart actually works smoothly. First impressions are permanent.
7. **Launching too late** - Don't wait for perfection. Ship when the core value prop works.
8. **Copying LangChain's API** - The market is saturated with LangChain-likes. Differentiate clearly.
9. **No tests** - Repos without CI/tests signal low quality to experienced developers.
10. **Choosing a bad license** - Use MIT or Apache 2.0. Anything more restrictive scares companies away.
11. **Marketing speak in README** - "Revolutionizing AI agent development" means nothing. Show code.
12. **Too many badges** - More than 6-8 badges looks cluttered.

---

## 11. Pre-Launch Checklist

### Repository
- [ ] Professional logo (SVG, centered in README)
- [ ] README follows the structure in Section 1
- [ ] LICENSE file (MIT or Apache 2.0)
- [ ] CONTRIBUTING.md with clear setup instructions
- [ ] CODE_OF_CONDUCT.md
- [ ] CHANGELOG.md
- [ ] .github/ templates (issues, PRs)
- [ ] CI/CD pipeline (GitHub Actions: test, lint, publish)
- [ ] Test suite with >80% coverage
- [ ] GitHub Topics set
- [ ] Repository description and website URL set

### Package
- [ ] Published on PyPI
- [ ] `pip install [name]` works cleanly
- [ ] Python 3.10+ support
- [ ] Minimal dependencies
- [ ] README example works when copy-pasted after install

### Documentation
- [ ] Documentation site live (MkDocs Material recommended)
- [ ] Getting Started guide (< 5 minutes)
- [ ] Core concepts explained
- [ ] API reference
- [ ] At least 5 working examples in examples/ directory

### Visuals
- [ ] Demo GIF or video (terminal recording)
- [ ] Architecture diagram
- [ ] Screenshots of output/results

### Content
- [ ] Blog post written and ready to publish
- [ ] HN "Show HN" post drafted
- [ ] Twitter launch thread drafted
- [ ] Reddit posts drafted (r/Python, r/MachineLearning, r/LocalLLaMA)

### Community
- [ ] Discord server created with channels
- [ ] GitHub Discussions enabled
- [ ] "good first issue" labels on 3-5 issues

### Amplification
- [ ] List of 20+ people to DM on launch day
- [ ] Newsletter submissions prepared
- [ ] Launch time scheduled (Tue-Thu, ~8-9 AM EST)

---

## 12. Post-Launch Week-by-Week Plan

### Week 1: Launch
- Day 1: Execute launch sequence (HN, Twitter, Reddit)
- Day 1-3: Respond to EVERY comment, issue, question
- Day 2: Share "Day 1 results" on Twitter
- Day 3: Submit to newsletters
- Day 4-5: Post on Dev.to, LinkedIn, secondary subreddits
- Day 7: First "what we learned from launch" blog post

### Week 2: Capitalize
- Fix all bugs reported during launch week
- Ship improvements requested by users
- Write tutorials based on common questions
- Engage with new community members on Discord
- Reach out to AI newsletters and podcasts

### Week 3-4: Content
- Write 2-3 tutorial blog posts
- Record video walkthrough for YouTube
- Create comparison guide vs. alternatives
- Add more examples to the repo
- Ship a minor release with improvements

### Month 2: Community
- Launch contributor program
- Create "awesome-[framework]" repo
- Submit to PyCon / meetup talk CFPs
- Build integrations with popular tools
- Publish benchmark results
- Ship v0.2 with most-requested features

### Month 3+: Sustain
- Regular release cadence (every 1-2 weeks)
- Monthly blog posts
- Community showcase events
- Enterprise/production case studies
- Consider GitHub Sponsors / Open Collective

---

## Key Takeaways

1. **The README is your landing page.** Invest 40% of your launch effort into making it perfect.
2. **Time to Hello World must be < 2 minutes.** If someone can't get a working result quickly, they won't star.
3. **Lead with a visual demo.** A GIF of your framework in action is worth 1000 words of documentation.
4. **Launch is a coordinated campaign, not a single post.** HN + Twitter + Reddit + newsletters, all on the same day.
5. **The first 48 hours determine trajectory.** Getting on GitHub Trending creates a self-reinforcing loop.
6. **Be visible and responsive.** The creator's presence in comments, issues, and social media is a major growth driver.
7. **Differentiate clearly.** "Why not LangChain?" is the first question everyone will ask. Have a sharp answer.
8. **Ride the wave.** Launch when there's industry excitement (new model release, AI news cycle, conference season).
9. **Build in public before launch.** Share the journey to build an audience before launch day.
10. **Keep shipping.** Sustained growth comes from regular releases, content, and community engagement.
