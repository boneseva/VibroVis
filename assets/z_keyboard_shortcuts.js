/**
 * keyboard_shortcuts.js
 * 
 * Global keyboard shortcuts for VibroVis:
 * - Ctrl + Space (Windows/Linux) or Cmd + Space (Mac) → Toggle audio playback
 * 
 * This runs entirely clientside with zero server latency.
 * The audio element maintains focus on the active input field.
 * 
 * CRITICAL: This listener is attached with HIGH specificity to avoid interfering
 * with tab-focus mechanisms, input event handlers, or other Dash callbacks.
 */
(function() {
    'use strict';

    // Guard against duplicate registration on React re-renders or multiple script loads
    if (window._vibrovisKeyboardShortcuts) {
        return;
    }
    window._vibrovisKeyboardShortcuts = true;

    /**
     * Attach the global keydown listener with proper error isolation
     * Only listen for Ctrl/Cmd + Space, ignore all other keys
     */
    document.addEventListener('keydown', function(event) {
        try {
            // Check ONLY for the specific modifier + space combination
            // Exit early for any other key to minimize overhead
            if (event.code !== 'Space') {
                return;
            }
            
            // Check for Ctrl (Windows/Linux) or Cmd (Mac)
            if (!event.ctrlKey && !event.metaKey) {
                return;
            }

            // At this point, we have Ctrl/Cmd + Space
            // Prevent browser's default behavior for this specific combo
            event.preventDefault();

            // Find the audio player element
            const audioPlayer = document.getElementById('audio-player');
            if (!audioPlayer) {
                console.warn('[VibroVis] Audio player element not found');
                return;
            }

            // Toggle playback state with error handling
            try {
                if (audioPlayer.paused) {
                    audioPlayer.play().catch(function(error) {
                        console.warn('[VibroVis] Failed to play audio:', error);
                    });
                } else {
                    audioPlayer.pause();
                }
            } catch (error) {
                console.error('[VibroVis] Error toggling audio playback:', error);
                return;
            }

            // Optional: Show visual feedback (subtle, non-intrusive)
            // This is purely visual and doesn't affect event propagation
            try {
                var feedback = document.getElementById('keyboard-shortcut-feedback');
                if (!feedback) {
                    feedback = document.createElement('div');
                    feedback.id = 'keyboard-shortcut-feedback';
                    feedback.style.cssText = 'position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.7);color:#fff;padding:8px 12px;border-radius:4px;font-size:12px;z-index:10000;pointer-events:none;opacity:0;transition:opacity 0.2s;';
                    document.body.appendChild(feedback);
                }

                feedback.textContent = audioPlayer.paused ? '⏸ Paused' : '▶ Playing';
                feedback.style.opacity = '1';
                
                if (feedback._fadeoutTimer) {
                    clearTimeout(feedback._fadeoutTimer);
                }
                
                feedback._fadeoutTimer = setTimeout(function() {
                    feedback.style.opacity = '0';
                }, 600);
            } catch (error) {
                // Feedback creation failed; this is non-critical
                console.warn('[VibroVis] Could not create feedback indicator:', error);
            }

        } catch (error) {
            // Outer error handler to ensure this listener never crashes the app
            console.error('[VibroVis] Keyboard shortcut handler error:', error);
        }
    }, false); // Use capturing phase = false (bubbling phase) to minimize interference
})();

