import random

moves = {
    "🗿": ["✂️", "🦎"],
    "📄": ["🗿", "🖖"],
    "✂️": ["📄", "🦎"],
    "🦎": ["📄", "🖖"],
    "🖖": ["🗿", "✂️"]
}

print("=== RPSLS ===")
print("Choose: 🗿 📄 ✂️ 🦎 🖖")
score = 0

while True:
    player = input("\nYour move (Q to quit): ").strip().lower()
    if player == "q":
        break
    if player not in moves:
        print("Invalid move! Copy emoji from list")
        continue
    
    computer = random.choice(list(moves.keys()))
    print(f"\nYOU: {player} vs CPU: {computer}")
    
    if computer in moves[player]:
        print("You win! 🎉")
        score += 1
    elif player == computer:
        print("Tie! 🤝")
    else:
        print("You lose! 💀")
    
    print(f"Score: {score}")

print(f"\nFinal score: {score}. Thanks for playing!")