document.addEventListener('DOMContentLoaded', () => {
    const calculateBtn = document.getElementById('calculate-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsPanel = document.getElementById('results-panel');
    const inputPanel = document.querySelector('.input-panel');

    // Bad and Unhealthy ingredient lists (lowercase for matching)
    const BAD_INGREDIENTS = [
        'high fructose corn syrup',
        'hydrogenated', // captures hydrogenated oils/trans fats
        'aspartame',
        'sucralose',
        'saccharin',
        'red 40',
        'yellow 5',
        'yellow 6',
        'blue 1',
        'bha',
        'bht',
        'sodium nitrate',
        'sodium nitrite',
        'potassium bromate',
        'titanium dioxide'
    ];

    const UNHEALTHY_INGREDIENTS = [
        'added sugar',
        'cane sugar',
        'sugar',
        'palm oil',
        'enriched flour',
        'refined wheat',
        'maltodextrin',
        'dextrose',
        'soybean oil',
        'canola oil',
        'corn syrup'
    ];

    calculateBtn.addEventListener('click', calculateScore);
    resetBtn.addEventListener('click', resetApp);

    function calculateScore() {
        // 1. Get Inputs
        const calories = parseFloat(document.getElementById('calories').value);
        const protein = parseFloat(document.getElementById('protein').value);
        const carbs = parseFloat(document.getElementById('carbs').value);
        const fat = parseFloat(document.getElementById('fat').value);
        const ingredientsText = document.getElementById('ingredients').value.toLowerCase();

        if (isNaN(calories) || isNaN(protein) || isNaN(carbs) || isNaN(fat)) {
            alert("Please enter valid numbers for all nutrition facts.");
            return;
        }

        let currentScore = 10;
        let macroPenalty = 0;
        let badPenalty = 0;
        let unhealthyPenalty = 0;

        // 2. Macro Calculation
        const calculatedCalories = (protein * 4) + (carbs * 4) + (fat * 9);
        const calorieDiff = Math.abs(calculatedCalories - calories);
        
        let macroMessage = `Calculated macros yield ~${Math.round(calculatedCalories)} kcal vs stated ${calories} kcal.`;
        let macroColor = 'neutral'; // default blue
        let macroBarWidth = '100%';

        // If stated calories differ by more than 20%, deduct points (indicates hidden macros or alcohol/sugar alcohols not accounted for properly)
        if (calories > 0 && (calorieDiff / calories) > 0.2) {
            macroPenalty = 2;
            macroMessage += " Significant deviation detected (-2 pts).";
            macroColor = 'danger';
            macroBarWidth = '80%';
        } else if (calories > 0 && (calorieDiff / calories) > 0.1) {
            macroPenalty = 1;
            macroMessage += " Minor deviation detected (-1 pt).";
            macroColor = 'warning';
            macroBarWidth = '90%';
        } else {
            macroMessage += " Macros perfectly match stated calories.";
            macroColor = 'accent'; // green
        }

        currentScore -= macroPenalty;

        // 3. Ingredients Scanning
        const foundBad = [];
        const foundUnhealthy = [];

        // Check Bad Ingredients
        BAD_INGREDIENTS.forEach(ingredient => {
            if (ingredientsText.includes(ingredient)) {
                foundBad.push(ingredient);
                badPenalty += 2;
            }
        });

        // Check Unhealthy Ingredients (Make sure we don't double count e.g. "high fructose corn syrup" and "corn syrup")
        UNHEALTHY_INGREDIENTS.forEach(ingredient => {
            // Only add if it's not a substring of an already found bad ingredient
            const isSubStringOfBad = foundBad.some(bad => bad.includes(ingredient));
            if (!isSubStringOfBad && ingredientsText.includes(ingredient)) {
                foundUnhealthy.push(ingredient);
                unhealthyPenalty += 1;
            }
        });

        currentScore -= (badPenalty + unhealthyPenalty);
        
        // Clamp score between 1 and 10
        const finalScore = Math.max(1, Math.min(10, currentScore));

        // 4. Update UI
        updateUI(finalScore, macroPenalty, macroMessage, macroColor, macroBarWidth, foundBad, foundUnhealthy, badPenalty, unhealthyPenalty);
    }

    function updateUI(score, macroPenalty, macroMessage, macroColor, macroBarWidth, badList, unhealthyList, badPenalty, unhealthyPenalty) {
        // Hide input, show results
        inputPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultsPanel.classList.add('animate-slide-in');

        // Update Score Circle
        const scoreElement = document.getElementById('final-score');
        const scoreCircle = document.querySelector('.score-circle');
        const messageElement = document.getElementById('score-message');

        // Animate counter
        let currentCount = 0;
        const duration = 1000;
        const interval = 20;
        const steps = duration / interval;
        const increment = score / steps;

        const counter = setInterval(() => {
            currentCount += increment;
            if (currentCount >= score) {
                currentCount = score;
                clearInterval(counter);
            }
            scoreElement.innerText = Math.round(currentCount);
        }, interval);

        // Determine colors based on final score
        let mainColor = '';
        if (score >= 8) {
            mainColor = 'var(--accent-color)';
            messageElement.innerText = 'Great Choice! Very Healthy.';
            messageElement.style.color = mainColor;
        } else if (score >= 5) {
            mainColor = 'var(--warning-color)';
            messageElement.innerText = 'Moderate. Consume in moderation.';
            messageElement.style.color = mainColor;
        } else {
            mainColor = 'var(--danger-color)';
            messageElement.innerText = 'Poor. Avoid if possible.';
            messageElement.style.color = mainColor;
        }

        // Set conic gradient for score circle
        const percentage = (score / 10) * 100;
        setTimeout(() => {
            scoreCircle.style.background = `conic-gradient(${mainColor} ${percentage}%, rgba(255,255,255,0.05) 0%)`;
            scoreCircle.style.boxShadow = `inset 0 0 20px rgba(0,0,0,0.5), 0 0 30px ${mainColor}40`;
        }, 100);

        // Update Macro Section
        const macroScoreBadge = document.getElementById('macro-score');
        const macroDetails = document.getElementById('macro-details');
        const macroBar = document.getElementById('macro-bar');

        macroScoreBadge.innerText = `Base: 10 (-${macroPenalty} pts)`;
        macroScoreBadge.className = `badge ${macroColor}`;
        macroDetails.innerText = macroMessage;
        
        let barColor = 'var(--primary-color)';
        if(macroColor === 'danger') barColor = 'var(--danger-color)';
        if(macroColor === 'warning') barColor = 'var(--warning-color)';
        if(macroColor === 'accent') barColor = 'var(--accent-color)';
        
        macroBar.style.width = '0%';
        macroBar.style.backgroundColor = barColor;
        setTimeout(() => { macroBar.style.width = macroBarWidth; }, 500);

        // Update Ingredients Lists
        const badContainer = document.getElementById('bad-ingredients-container');
        const badUl = document.getElementById('bad-list');
        const badScoreBadge = document.getElementById('bad-score');

        const unhealthyContainer = document.getElementById('unhealthy-ingredients-container');
        const unhealthyUl = document.getElementById('unhealthy-list');
        const unhealthyScoreBadge = document.getElementById('unhealthy-score');

        badUl.innerHTML = '';
        if (badList.length > 0) {
            badContainer.style.display = 'block';
            badScoreBadge.innerText = `-${badPenalty} pts`;
            badList.forEach(item => {
                const li = document.createElement('li');
                li.innerText = item.charAt(0).toUpperCase() + item.slice(1);
                badUl.appendChild(li);
            });
        } else {
            badContainer.style.display = 'none';
        }

        unhealthyUl.innerHTML = '';
        if (unhealthyList.length > 0) {
            unhealthyContainer.style.display = 'block';
            unhealthyUl.className = 'ingredient-list warning';
            unhealthyScoreBadge.innerText = `-${unhealthyPenalty} pts`;
            unhealthyList.forEach(item => {
                const li = document.createElement('li');
                li.innerText = item.charAt(0).toUpperCase() + item.slice(1);
                unhealthyUl.appendChild(li);
            });
        } else {
            unhealthyContainer.style.display = 'none';
        }
    }

    function resetApp() {
        resultsPanel.style.display = 'none';
        inputPanel.style.display = 'block';
        
        // Reset score circle
        const scoreCircle = document.querySelector('.score-circle');
        scoreCircle.style.background = `conic-gradient(var(--accent-color) 0%, transparent 0%)`;
        
        // Don't reset inputs, allow user to tweak them
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
