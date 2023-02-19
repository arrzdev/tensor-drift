from agent import Agent

def main():
  #little gui to chosse between record, train and play
  agent = Agent()

  d = input("record, train or play? (r/t/p) ")
  if d == "r":
    agent.record()
  elif d == "t":
    agent.train()
  elif d == "p":
    agent.play()
  else:
    print("WRONG INPUT")

if __name__ == "__main__":
  main()