import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

const docsBase = "/reinforcement-learning-lab/";
const siteUrl  = "https://dash10107.github.io/reinforcement-learning-lab";
const githubRepoLink = "https://github.com/Dash10107/rl-portfolio";
const huggingFaceLink = "https://huggingface.co/Dash10107";

// ─── Sidebar ──────────────────────────────────────────────────────────────────

const enChapterItems = [
  { text: "Welcome", link: "/en/" },
  { text: "The MDP Framework", link: "/en/modules/mdp/" },

  {
    text: "Part 1 — Foundations",
    items: [
      { text: "What Is an Agent?", link: "/en/modules/what-is-rl/" },
      { text: "The Problem With Guessing", link: "/en/modules/bandits/" },
      { text: "Smarter Ways to Explore", link: "/en/modules/exploration/" },
    ],
  },
  {
    text: "Part 2 — Memory",
    items: [
      { text: "Teaching an Agent to Remember", link: "/en/modules/q-learning/" },
      { text: "Waiting for the Ending", link: "/en/modules/monte-carlo/" },
      { text: "Temporal Difference Learning", link: "/en/modules/td-learning/" },
    ],
  },
  {
    text: "Part 3 — Scale",
    items: [
      { text: "When the Maze Gets Too Big", link: "/en/modules/dqn/" },
    ],
  },
  {
    text: "Part 4 — Intention",
    items: [
      { text: "Why Learn the Policy Directly?", link: "/en/modules/policy-gradient-math/" },
      { text: "Two Brains Are Better Than One", link: "/en/modules/a2c/" },
      { text: "PPO — The Algorithm Behind Modern AI", link: "/en/modules/ppo/" },
      { text: "Continuous Control (SAC)", link: "/en/modules/sac/" },
    ],
  },
  {
    text: "Part 5 — Together",
    items: [
      { text: "What Happens With Two of You?", link: "/en/modules/multi-agent-rl/" },
      { text: "When No One Planned It", link: "/en/modules/swarm/" },
    ],
  },
  {
    text: "Part 6 — Imagination",
    items: [
      { text: "What If the Agent Could Imagine?", link: "/en/modules/mbrl/" },
    ],
  },
  {
    text: "Part 7 — Alignment",
    items: [
      { text: "Teaching AI What You Actually Want", link: "/en/modules/rlhf/" },
    ],
  },
  {
    text: "Bonus",
    items: [
      { text: "When the World Has Moods (HMM)", link: "/en/modules/hmm/" },
      { text: "When the World Has Gravity (Unity)", link: "/en/modules/unity/" },
    ],
  },
];

const enDemoItems = [
  { text: "All Live Demos", link: "/en/demos/" },
  { text: "Maze Solver", link: "/en/demos/maze-solver/" },
  { text: "Rocket Lander (SAC)", link: "/en/demos/rocket-lander/" },
  { text: "Warehouse Robots", link: "/en/demos/warehouse/" },
  { text: "Calligrapher RLHF", link: "/en/demos/calligrapher/" },
];

const enResourceItems = [
  { text: "Overview", link: "/en/resources/" },
  { text: "Core Concepts", link: "/en/resources/concepts/" },
  { text: "Getting Started", link: "/en/resources/getting-started/" },
  { text: "Starter Templates", link: "/en/resources/starter-templates/" },
  { text: "Roadmap", link: "/en/resources/roadmap/" },
];

// ─── Config ───────────────────────────────────────────────────────────────────

export default withMermaid(
  defineConfig({
    base: docsBase,
    title: "Reinforcement Learning Lab",
    description:
      "Learn Reinforcement Learning by Building — from Q-Learning to PPO, SAC, Multi-Agent RL and RLHF. Every chapter has a live demo you can run right now.",

    head: [
      ["link", { rel: "preconnect", href: "https://fonts.googleapis.com" }],
      ["link", { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" }],
      [
        "link",
        {
          rel: "stylesheet",
          href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap",
        },
      ],
      // ── Favicon & theme colour ──────────────────────────────────
      ["meta", { name: "theme-color", content: "#0D7A6E" }],
      ["link", { rel: "icon", type: "image/svg+xml", href: `${docsBase}favicon.svg` }],
      ["link", { rel: "canonical", href: siteUrl }],
      ["link", { rel: "preconnect", href: "https://huggingface.co" }],
      // ── Primary SEO meta ────────────────────────────────────────
      ["meta", { name: "author", content: "Daksh Jain" }],
      ["meta", { name: "robots", content: "index, follow" }],
      [
        "meta",
        {
          name: "keywords",
          content: [
            "reinforcement learning tutorial",
            "reinforcement learning course",
            "reinforcement learning from scratch",
            "learn reinforcement learning",
            "deep reinforcement learning",
            "Q learning tutorial",
            "Q learning python",
            "DQN explained",
            "deep Q network tutorial",
            "PPO reinforcement learning",
            "proximal policy optimization",
            "actor critic reinforcement learning",
            "A2C reinforcement learning",
            "SAC reinforcement learning",
            "soft actor critic",
            "RLHF explained",
            "reinforcement learning from human feedback",
            "multi agent reinforcement learning",
            "policy gradient explained",
            "Bellman equation explained",
            "TD learning reinforcement learning",
            "temporal difference learning",
            "Monte Carlo reinforcement learning",
            "model based reinforcement learning",
            "reinforcement learning beginner",
            "reinforcement learning intermediate",
            "reinforcement learning python",
            "reinforcement learning pytorch",
            "gymnasium reinforcement learning",
            "openai gym reinforcement learning",
            "MDP markov decision process",
            "reward function reinforcement learning",
            "exploration exploitation tradeoff",
            "UCB upper confidence bound",
            "Thompson sampling bandit",
            "IPPO multi agent",
            "QMIX cooperative MARL",
            "DPO direct preference optimization",
          ].join(", "),
        },
      ],
      // ── Open Graph ─────────────────────────────────────────────
      ["meta", { property: "og:type", content: "website" }],
      ["meta", { property: "og:site_name", content: "Reinforcement Learning Lab" }],
      ["meta", { property: "og:title", content: "Reinforcement Learning Lab — Learn RL from Scratch" }],
      [
        "meta",
        {
          property: "og:description",
          content:
            "Free, open-source RL course covering Q-Learning, DQN, PPO, SAC, Multi-Agent RL, and RLHF. Real math, runnable code, beginner to intermediate.",
        },
      ],
      ["meta", { property: "og:url", content: siteUrl }],
      ["meta", { property: "og:image", content: `${siteUrl}/og-image.png` }],
      // ── Twitter / X Card ───────────────────────────────────────
      ["meta", { name: "twitter:card", content: "summary_large_image" }],
      ["meta", { name: "twitter:title", content: "Reinforcement Learning Lab — Learn RL from Scratch" }],
      [
        "meta",
        {
          name: "twitter:description",
          content:
            "Free RL course: Q-Learning → DQN → PPO → SAC → RLHF. Real math, clean code, beginner to intermediate.",
        },
      ],
      ["meta", { name: "twitter:image", content: `${siteUrl}/og-image.png` }],
      // ── JSON-LD Structured Data (Course schema) ─────────────────
      [
        "script",
        { type: "application/ld+json" },
        JSON.stringify({
          "@context": "https://schema.org",
          "@type": "Course",
          name: "Reinforcement Learning Lab",
          description:
            "A free, open-source course on reinforcement learning covering Q-Learning, DQN, PPO, SAC, Multi-Agent RL, RLHF, and more. Designed for beginners and intermediate learners.",
          url: siteUrl,
          provider: {
            "@type": "Person",
            name: "Daksh Jain",
            sameAs: [githubRepoLink, huggingFaceLink],
          },
          educationalLevel: "Beginner to Intermediate",
          teaches: [
            "Reinforcement Learning",
            "Q-Learning",
            "Deep Q-Networks",
            "Policy Gradient Methods",
            "Proximal Policy Optimization",
            "Soft Actor-Critic",
            "Multi-Agent Reinforcement Learning",
            "Reinforcement Learning from Human Feedback",
            "Model-Based Reinforcement Learning",
          ],
          isAccessibleForFree: true,
          inLanguage: "en",
          hasCourseInstance: {
            "@type": "CourseInstance",
            courseMode: "online",
            courseWorkload: "PT15H",
          },
        }),
      ],
    ],

    cleanUrls: true,

    markdown: {
      math: true,
    },

    themeConfig: {
      logo: {
        light: "/logo-light.svg",
        dark: "/logo-dark.svg",
        alt: "RL Lab",
      },
      siteTitle: "RL Lab",

      nav: [
        {
          text: "Chapters",
          link: "/en/modules/what-is-rl/",
          activeMatch: "^/en/(modules/|)($|.*)",
        },
        {
          text: "Live Demos",
          link: "/en/demos/",
          activeMatch: "^/en/demos/",
        },
        {
          text: "Resources",
          link: "/en/resources/",
          activeMatch: "^/en/resources/",
        },
        {
          text: "▶ Try Demo ↗",
          link: huggingFaceLink,
        },
      ],

      sidebar: {
        "/en/modules/": enChapterItems,
        "/en/demos/":   enDemoItems,
        "/en/resources/": enResourceItems,
        "/en/": enChapterItems,
      },

      socialLinks: [
        { icon: "github", link: githubRepoLink },
      ],

      footer: {
        message: "Released under the MIT License.",
        copyright: "© 2025 Reinforcement Learning Lab · Built with ❤️ and PyTorch",
      },

      editLink: {
        pattern: `${githubRepoLink}/edit/main/site/docs/:path`,
        text: "Edit this page on GitHub",
      },

      search: { provider: "local" },

      outline: {
        level: [2, 3],
        label: "On this page",
      },

      lastUpdated: { text: "Last updated" },
    },

    mermaid: { theme: "neutral" },

    sitemap: {
      hostname: siteUrl,
    },
  })
);
