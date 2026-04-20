import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add Splash Screen CSS
splash_css = """
    /* ===================== NEW UI CHANGES ===================== */
    /* Splash Screen */
    #splash-screen {
      position: fixed; inset: 0; z-index: 10000;
      background: var(--bg); display: flex; align-items: center; justify-content: center;
      transition: transform 0.8s cubic-bezier(0.85, 0, 0.15, 1);
    }
    #splash-screen.swipe-left { transform: translateX(-100%); }
    .splash-logo {
      font-family: 'DM Serif Display', serif; font-size: 5rem;
      letter-spacing: -0.02em; color: var(--text);
    }
    .splash-logo span { color: var(--gold); font-style: italic; }

    /* Marketing Section */
    .marketing-section {
      width: 100%; max-width: 1000px; margin: 4rem auto; padding: 0 1.5rem;
      display: flex; flex-direction: column; gap: 4rem; z-index: 2; position: relative;
    }
    .hero-video-container {
      width: 100%; border-radius: 20px; overflow: hidden;
      border: 1px solid var(--border2);
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    .hero-video-container video { width: 100%; display: block; }
    
    .feature-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;
    }
    .feature-img-container {
      border-radius: 20px; overflow: hidden; border: 1px solid var(--border2);
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    .feature-img-container img { width: 100%; display: block; }
    .feature-text h2 { font-family: 'DM Serif Display', serif; font-size: 2.5rem; margin-bottom: 1rem; color: var(--text); }
    .feature-text p { font-size: 1.1rem; color: var(--text-2); line-height: 1.6; }
    
    /* Reviews Section */
    .reviews-section {
      text-align: center;
    }
    .reviews-section h2 { font-family: 'DM Serif Display', serif; font-size: 2.5rem; margin-bottom: 2rem; }
    .reviews-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;
    }
    .review-card {
      background: rgba(31, 41, 55, 0.6); border: 1px solid var(--border2);
      border-radius: 16px; padding: 1.5rem; text-align: left;
      backdrop-filter: blur(12px);
    }
    .review-stars { color: #F59E0B; font-size: 1.2rem; margin-bottom: 0.5rem; }
    .review-text { font-size: 0.95rem; color: var(--text-2); line-height: 1.5; margin-bottom: 1rem; font-style: italic; }
    .review-author { display: flex; align-items: center; gap: 0.75rem; }
    .review-avatar { width: 40px; height: 40px; border-radius: 50%; background: var(--gold); display: flex; align-items: center; justify-content: center; font-weight: bold; color: #fff; }
    .review-name { font-weight: 600; color: var(--text); }

    /* Wide Chat Container */
    .chat-container.wide { max-width: 900px !important; }

    @media (max-width: 768px) {
      .feature-grid { grid-template-columns: 1fr; }
      .feature-text h2 { font-size: 2rem; }
      .splash-logo { font-size: 3.5rem; }
      .chat-container.wide { max-width: 95vw !important; }
    }
"""

if "/* ===================== NEW UI CHANGES ===================== */" not in content:
    content = content.replace("</style>", splash_css + "\n  </style>")

# 2. Add Splash Screen HTML
splash_html = """
<!-- Splash Screen -->
<div id="splash-screen">
  <div class="splash-logo">LEARN<span>AI</span></div>
</div>

<script>
  window.addEventListener('load', () => {
    setTimeout(() => {
      document.getElementById('splash-screen').classList.add('swipe-left');
      setTimeout(() => {
        document.getElementById('splash-screen').style.display = 'none';
      }, 800);
    }, 1500); // 1.5s delay
  });
</script>
"""

if "id=\"splash-screen\"" not in content:
    content = content.replace("<body>", "<body>\n" + splash_html)

# 3. Change EduPro to LEARNAI
content = content.replace("Edu<span>Pro</span>", "LEARN<span>AI</span>")

# 4. Adjust the main container style
# Change max-width of chat-container
content = content.replace('class="chat-container" style="max-width: 600px;', 'class="chat-container wide" style="')
# And if it doesn't have style inline
content = content.replace('class="chat-container"', 'class="chat-container wide"')
content = content.replace('class="chat-container wide" wide"', 'class="chat-container wide"') # fix double replacement

# 5. Add Homepage content below chat-box
marketing_html = """
    <!-- Marketing Section -->
    <div class="marketing-section">
      
      <!-- Video Section -->
      <div class="hero-video-container">
        <video autoplay loop muted playsinline>
          <source src="../googlellm.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>

      <!-- Feature Section -->
      <div class="feature-grid">
        <div class="feature-text">
          <h2>Deep Understanding, Instantly</h2>
          <p>Transform any topic into a rich, interactive learning experience. Similar to NotebookLM, LEARNAI synthesizes vast amounts of information and presents it in a clear, digestible format tailored to your learning style.</p>
        </div>
        <div class="feature-img-container">
          <img src="../googlellmimage.png" alt="LEARNAI Feature Interface">
        </div>
      </div>

      <!-- Reviews Section -->
      <div class="reviews-section">
        <h2>Loved by Learners</h2>
        <div class="reviews-grid">
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"LEARNAI completely changed how I study for my engineering exams. The deep dive feature is absolutely phenomenal!"</div>
            <div class="review-author">
              <div class="review-avatar">R</div>
              <div class="review-name">Rahul Sharma</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"The interface feels incredibly professional and smooth. It's like having a personal tutor available 24/7."</div>
            <div class="review-author">
              <div class="review-avatar">P</div>
              <div class="review-name">Priya Patel</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"I love how it breaks down complex medical concepts into simple, visual components. Highly recommended!"</div>
            <div class="review-author">
              <div class="review-avatar">A</div>
              <div class="review-name">Amit Kumar</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"The speed and quality of responses are unmatched. Better than ChatGPT for focused learning."</div>
            <div class="review-author">
              <div class="review-avatar">S</div>
              <div class="review-name">Sneha Gupta</div>
            </div>
          </div>
        </div>
      </div>

    </div>
"""

# Find the end of the center content area where the chat box is.
# Let's insert the marketing_html before the final closing div of screen-start
# Wait, screen-start has:
# <div id="screen-start" ...>
#   <div ad-sidebar ...>
#   <div class="start-glow-blob">...
#   <div style="text-align:center; ... width:100%; ... padding: 1.5rem;">
#      ...
#      <div class="chat-container">...</div>
#      ... (feature pills, btn-primary)
#   </div>
# </div>

# The closing </div> for the text-align:center div is followed by the closing </div> for screen-start.
# It might be safer to replace '<div class="chat-container wide" style="width: 100%;">' or just regex.
# Let's write the modified content back.

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied basic edits.")
