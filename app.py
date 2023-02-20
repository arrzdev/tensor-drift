from agent import Agent

def main():
  #little gui to chosse between record, train and play
  agent = Agent()

  d = input("record, pre-process, train or run? (r/pp/t/p) ")
  if d == "r":
    agent.record()
  elif d == "t":
    agent.train()
  elif d == "p":
    agent.play()
  elif d == "pp":
    agent.pre_process()
  else:
    print("WRONG INPUT")

if __name__ == "__main__":
  main()